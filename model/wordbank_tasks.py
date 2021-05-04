from dataset_orm import OriginalDataset, Sentence
import numpy as np
import pandas as pd
import tqdm
import warnings
from dataset_orm import *


def select_k_random(items, k):
    return [items[i] for i in np.random.permutation(len(items))[:k]]

def select_k_random_n_times(items, k, n):
    return [select_k_random(items, k) for _ in range(n)]


def smallest_nll_criterion(scores):
    if scores and len(scores) > 0:
        return np.argmax(scores) == 0    
    return False


def discriminative_task_single_word(
    session_maker, target_wordbank_word, 
    n_sentences_per_word, n_alternative_words, 
    model_names, model_scorers, criterion_func, 
    random_seed=33, same_category_words=True,
    original_dataset=None):

    np.random.seed(random_seed)
    session = session_maker()

    sentence_query = session.query(Sentence.id, Sentence.text).\
        filter(Sentence.wordbank_words.any(id=target_wordbank_word.id))

    if original_dataset is not None:
        sentence_query = sentence_query.filter(Sentence.original_dataset == original_dataset)

    word_sentences = sentence_query.all()
    if len(word_sentences) < n_sentences_per_word:
        warnings.warn(f'For word {target_wordbank_word.word} (id {target_wordbank_word.id}), only found {len(word_sentences)} sentences, fewer than the requested {n_sentences_per_word}. Skipping...', UserWarning)
        return []
    
    ids_and_sentences = select_k_random(word_sentences, n_sentences_per_word)
    sentence_ids, sentences = zip(*ids_and_sentences)

    word_query = session.query(WordbankWord.id, WordbankWord.word)
    if same_category_words:
        word_query = word_query.filter(WordbankWord.category == target_wordbank_word.category).filter(WordbankWord.id != target_wordbank_word.id)
    else:
        word_query = word_query.filter(WordbankWord.category != target_wordbank_word.category)
    ids_and_words_per_sentence = select_k_random_n_times(word_query.all(), n_alternative_words, n_sentences_per_word)
    word_ids_per_sentence, words_per_sentence = list(zip(*[list(zip(*x)) for x in ids_and_words_per_sentence]))

    word_ids_per_sentence = [list(word_ids) for word_ids in word_ids_per_sentence]
    [word_ids.insert(0, target_wordbank_word.id) for word_ids in word_ids_per_sentence]
    words_per_sentence = [list(words) for words in words_per_sentence]
    [words.insert(0, target_wordbank_word.word) for words in words_per_sentence]

    sentence_copies = [s.replace(target_wordbank_word.word, w, 1) 
                        for s, words in zip(sentences, words_per_sentence)
                        for w in words]
    model_sentence_scores = [scorer.score_sentences(sentence_copies) for scorer in model_scorers]
    all_results = []

    for model_name, model_raw_scores in zip(model_names, model_sentence_scores):
        for s, (sentence_id, sentence_text) in enumerate(ids_and_sentences):
            sentence_scores = model_raw_scores[s * n_alternative_words:(s + 1) * n_alternative_words]
            result = criterion_func(sentence_scores)
        
            all_results.append(dict(
                model_name=model_name, 
                sentence_id=sentence_id, 
                sentence_text=sentence_text, 
                compared_word_ids=word_ids_per_sentence[s],
                compared_words=words_per_sentence[s],
                sentence_scores=sentence_scores,
                result=result 
            ))

    return all_results


def discriminative_task_all_words(session_maker, 
    n_sentences_per_word, n_alternative_words, 
    model_names, model_scorers, criterion_func, 
    random_seed=33, same_category_words=True,
    original_dataset=None):

    np.random.seed(random_seed)
    session = session_maker()

    if original_dataset is not None:
        if isinstance(original_dataset, str):
            dataset_obj = session.query(OriginalDataset).filter(OriginalDataset.name == original_dataset).one_or_none()
            if dataset_obj is None:
                raise ValueError(f'Received original dataset string "{original_dataset}", which does not exist in DB. Aborting.')
            
            original_dataset = dataset_obj

    all_words = session.query(WordbankWord).all()
    words_without_spaces = [w for w in all_words if w.word.count(' ') == 0]

    all_results = []
    for target_word in tqdm.tqdm(words_without_spaces, total=len(words_without_spaces)):
        target_word_results = discriminative_task_single_word(
            session_maker=session_maker, target_wordbank_word=target_word,
            n_sentences_per_word=n_sentences_per_word, n_alternative_words=n_alternative_words, 
            model_names=model_names, model_scorers=model_scorers, criterion_func=criterion_func, 
            random_seed=random_seed, same_category_words=same_category_words, original_dataset=original_dataset)

        for result in target_word_results:
            result['word_id'] = target_word.id
            result['word'] = target_word.word

        all_results.extend(target_word_results)

    return pd.DataFrame.from_records(all_results)


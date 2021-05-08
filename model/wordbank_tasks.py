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
    result = False
    if scores and len(scores) > 0:
        result = np.argmax(scores) == 0    
    return dict(result=result)


def find_rank_of_first(scores, threshold=0.5):
    n = len(scores)
    temp = scores.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(n)
    r = n - 1 - ranks[0]
    p = (r + 1) / n
    return dict(rank=r, percentile=p, result=p < threshold)


def discriminative_task_single_word(
    session_maker, target_wordbank_word, 
    n_sentences_per_word, n_alternative_words, 
    scorer, criterion_func, 
    batch_size=256, random_seed=33, same_category_words=True,
    original_dataset=None, criterion_func_kwargs=None):

    np.random.seed(random_seed)
    session = session_maker()

    if criterion_func_kwargs is None:
        criterion_func_kwargs = dict()

    sentence_query = session.query(Sentence.id, Sentence.text).\
        filter(Sentence.wordbank_words.any(id=target_wordbank_word.id))

    if original_dataset is not None:
        sentence_query = sentence_query.filter(Sentence.original_dataset == original_dataset)

    word_sentences = sentence_query.all()
    
    if len(word_sentences) == 0:
        warnings.warn(f'For word {target_wordbank_word.word} (id {target_wordbank_word.id}), there are no sentences...', UserWarning)
        return []
    
    if n_sentences_per_word == -1 or n_sentences_per_word == 'all':
        n_sentences_per_word = len(word_sentences)
        sentence_ids, sentences = zip(*word_sentences)

    else:
        if len(word_sentences) < n_sentences_per_word:
            warnings.warn(f'For word {target_wordbank_word.word} (id {target_wordbank_word.id}), only found {len(word_sentences)} sentences, fewer than the requested {n_sentences_per_word}.', UserWarning)
            n_sentences_per_word = len(word_sentences)
    
        ids_and_sentences = select_k_random(word_sentences, n_sentences_per_word)
        sentence_ids, sentences = zip(*ids_and_sentences)

    word_query = session.query(WordbankWord.id, WordbankWord.word)
    if same_category_words:
        word_query = word_query.filter(WordbankWord.category == target_wordbank_word.category).filter(WordbankWord.id != target_wordbank_word.id)
    else:
        word_query = word_query.filter(WordbankWord.category != target_wordbank_word.category)

    if n_alternative_words == -1 or n_alternative_words == 'all':
        if not same_category_words:
            raise ValueError(f'Running on all words is only supported when running with same category words')

        word_ids, words = zip(*word_query.all())
        word_ids = list(word_ids)
        word_ids_per_sentence = [word_ids.copy() for _ in range(len(sentences))]
        words = list(words)
        words_per_sentence = [words.copy() for _ in range(len(sentences))]
    else:
        ids_and_words_per_sentence = select_k_random_n_times(word_query.all(), n_alternative_words, n_sentences_per_word)
        word_ids_per_sentence, words_per_sentence = list(zip(*[list(zip(*x)) for x in ids_and_words_per_sentence]))
        word_ids_per_sentence = [list(word_ids) for word_ids in word_ids_per_sentence]
        words_per_sentence = [list(words) for words in words_per_sentence]


    [word_ids.insert(0, target_wordbank_word.id) for word_ids in word_ids_per_sentence]
    [words.insert(0, target_wordbank_word.word) for words in words_per_sentence]

    sentence_copies = [[s.replace(target_wordbank_word.word, w, 1) for w in words]
                        for s, words in zip(sentences, words_per_sentence)]

    n_sentences_per_batch = batch_size // n_alternative_words
    sentence_scores = []
    for batch_idx in range(int(np.ceil(n_sentences_per_word) / n_sentences_per_batch)):
        batch_sentence_copies = sentence_copies[batch_idx * n_sentences_per_batch:min(n_sentences_per_word, (batch_idx + 1) * n_sentences_per_batch)]
        batch = [item for sublist in batch_sentence_copies for item in sublist]
        sentence_scores.extend(scorer.score_sentences(batch))

    all_results = []

    for s, (sentence_id, sentence_text) in enumerate(zip(sentence_ids, sentences)):
        sentence_scores = sentence_scores[s * n_alternative_words:(s + 1) * n_alternative_words]
        criterion_dict = criterion_func(sentence_scores, **criterion_func_kwargs)
        sentence_dict = dict(
            sentence_id=sentence_id, 
            sentence_text=sentence_text, 
            compared_word_ids=word_ids_per_sentence[s],
            compared_words=words_per_sentence[s],
            sentence_scores=sentence_scores,
            
        )
        sentence_dict.update(criterion_dict)
        all_results.append(sentence_dict)

    return all_results


def discriminative_task_all_words(session_maker, 
    n_sentences_per_word, n_alternative_words, 
    model_name, scorer, criterion_func, 
    random_seed=33, same_category_words=True,
    original_dataset=None, criterion_func_kwargs=None):

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
            scorer=scorer, criterion_func=criterion_func, 
            random_seed=random_seed, same_category_words=same_category_words, 
            original_dataset=original_dataset, criterion_func_kwargs=criterion_func_kwargs)

        for result in target_word_results:
            result['model_name'] = model_name
            result['word_id'] = target_word.id
            result['word'] = target_word.word

        all_results.extend(target_word_results)

    return pd.DataFrame.from_records(all_results)


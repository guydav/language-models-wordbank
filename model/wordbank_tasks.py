import numpy as np
from dataset_orm import *


def select_k_random(items, k):
    return [items[i] for i in np.random.permutation(len(items))[:k]]


def smallest_nll_criterion(scores):
    return np.argmax(scores) == 0    


def discriminative_task_single_word(
    session_maker, target_wordbank_word, 
    n_sentences, n_words, 
    model_scorers, criterion_func, 
    random_seed=33, same_category_words=True):

    np.random.seed(random_seed)
    session = session_maker()

    word_sentences = session.query(Sentence.text).\
        filter(Sentence.wordbank_words.any(id=target_wordbank_word.id)).\
        all()
    sentences = select_k_random([ws[0] for ws in word_sentences], n_sentences)
    word_query = session.query(WordbankWord.word)
    if same_category_words:
        word_query = word_query.filter(WordbankWord.category == target_wordbank_word.category)
    else:
        word_query = word_query.filter(WordbankWord.category != target_wordbank_word.category)
    words = select_k_random([wq[0] for wq in word_query.all()], n_words)

    words.insert(0, target_wordbank_word.word)
    sentence_copies = [s.replace(target_wordbank_word.word, w, 1) for s in sentences for w in words]
    model_sentence_scores = [scorer.score_sentences(sentence_copies) for scorer in model_scorers]
    per_model_scores = [[criterion_func(model_scores[s * n_words:(s + 1) * n_words]) for s in range(n_sentences)] for model_scores in model_sentence_scores]
    return per_model_scores


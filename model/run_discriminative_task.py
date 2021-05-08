import argparse
import os
import pathlib
import warnings

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import torch
import mxnet as mx
from transformers import AutoTokenizer, AutoModelForMaskedLM
from mlm.models import get_pretrained
from mlm.scorers import MLMScorerPT 
from wordbank_tasks import discriminative_task_all_words, find_rank_of_first


DB_FILE = 'wordbank.db'
DB_PATH = pathlib.Path(os.getcwd()).parent.absolute() / 'data' / DB_FILE



parser = argparse.ArgumentParser()

parser.add_argument('-c', '--checkpoint-name', required=True)
parser.add_argument('-o', '--output-folder', default=None)
DEFAULT_SENTENCES_PER_WORD = 'all'
parser.add_argument('-s', '--sentences-per-word', default=DEFAULT_SENTENCES_PER_WORD)
DEFAULT_ALTERNATIVE_WORDS = 'all'
parser.add_argument('-w', '--alternative-words', default=DEFAULT_ALTERNATIVE_WORDS)
DEFAULT_RANDOM_SEED = 33
parser.add_argument('-r', '--random-seed', default=DEFAULT_RANDOM_SEED)
parser.add_argument('-d', '--original-dataset', default=None)
parser.add_argument('--different-category-alternative-words', action='store_true')
DEFAULT_THRESHOLD = 0.5
parser.add_argument('-t', '--threshold', default=DEFAULT_THRESHOLD, help='Threshold for criterion func')


def scorer_from_transformers_checkpoint(checkpoint_name, contexts, device):
    try:
        model, vocab, tokenizer = get_pretrained(ctxs=contexts, name=checkpoint_name)
    except ValueError as e:
        print(f'mlm.models.get_pretrained failed, defaulting to Transformers: {e.args}')
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_name)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        vocab = None

    return MLMScorerPT(model, vocab, tokenizer, ctxs=contexts, device=device)


def main(args):
    contexts = [mx.gpu(0) if torch.cuda.is_available() else mx.cpu()]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scorer = scorer_from_transformers_checkpoint(args.checkpoint_name, contexts, device)
    engine = create_engine(f'sqlite:///{DB_PATH}')
    Session = sessionmaker(bind=engine)
    
    warnings.filterwarnings('ignore', category=UserWarning, module='gluonnlp.data')

    if args.sentences_per_word != 'all':
        args.sentences_per_word = int(args.sentences_per_word)

    if args.alternative_words != 'all':
        args.alternative_words = int(args.alternative_words)

    results_df = discriminative_task_all_words(
        session_maker=Session, n_sentences_per_word=args.sentences_per_word,
        n_alternative_words=args.alternative_words, model_name=args.checkpoint_name,
        scorer=scorer, criterion_func=find_rank_of_first,
        random_seed=args.random_seed, same_category_words=not args.different_category_alternative_words,
        original_dataset=args.original_dataset, criterion_func_kwargs=dict(threshold=args.threshold))

    name = args.checkpoint_name.replace('/', '_')
    if args.output_folder is None:
        args.output_folder = '.'

    output_file = f'{name}_sentences-{args.sentences_per_word}_words-{args.alternative_words}_seed-{args.random_seed}'
    output_file += f'_{args.different_category_alternative_words and "diff" or "same"}-category-words_{args.original_dataset is not None and args.original_dataset or "both-datasets"}.csv'

    results_df.to_csv(pathlib.Path(args.output_folder).absolute() / output_file)
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

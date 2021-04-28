import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from transformers import AutoTokenizer, AutoModelForMaskedLM
from mlm.scorers import MLMScorerPT 
from wordbank_tasks import discriminative_task_all_words, smallest_nll_criterion


DB_FILE = 'wordbank.db'
DB_PATH = pathlib.Path(os.getcwd()).parent.absolute() / 'data' / DB_FILE



parser = argparse.ArgumentParser()

parser.add_argument('-c', '--checkpoint-name', required=True)
parser.add_argument('-o', '--output-file', default=None)
DEFAULT_SENTENCES_PER_WORD = 10
parser.add_argument('-s', '--sentences-per-word', default=DEFAULT_SENTENCES_PER_WORD)
DEFAULT_ALTERNATIVE_WORDS = 10
parser.add_argument('-w', '--alternative-words', default=DEFAULT_ALTERNATIVE_WORDS)
DEFAULT_RANDOM_SEED = 33
parser.add_argument('-r', '--random-seed', default=DEFAULT_RANDOM_SEED)
parser.add_argument('-d', '--original-dataset', default=None)
parser.add_argument('--different-category-alternative-words', action='store_true')


def scorer_from_transformers_checkpoint(checkpotint_name):
    tokenizer = AutoTokenizer.from_pretrained(checkpotint_name)
    model = AutoModelForMaskedLM.from_pretrained(checkpotint_name)
    return MLMScorerPT(model, None, tokenizer, CONTEXTS, device=DEVICE)


def main(args):
    scorer = scorer_from_transformers_checkpoint(args.checkpoint_name)
    engine = create_engine(f'sqlite:///{DB_PATH}')
    Session = sessionmaker(bind=engine)
    
    results_df = discriminative_task_all_words(
        session_maker=Session, n_sentences_per_word=args.sentences_per_word,
        n_alternative_words=args.alternative_words, model_names=(args.checkpoint_name,),
        model_scorers=(scorer,), criterion_func=smallest_nll_criterion,
        random_seed=args.random_seed, same_category_words=not args.different_category_alternative_words,
        original_dataset=args.original_dataset)

    if args.output_file is None:
        name = args.checkpoint_name.replace('/', '_')
        args.output_file = f'{name}_sentences-{args.sentences_per_word}_words-{args.alternative_words}_seed-{args.random_seed}'
        args.output_file += f'_{args.different_category_alternative_words and "diff" or "same"}-category-words_{args.original_dataset is not None and args.original_dataset or "both-datasets"}.csv'

    results_df.to_csv(args.output_file)
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
import argparse
import pathlib
import glob
import numpy as np
import pandas as pd
import tqdm
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--input-folder', required=True)
parser.add_argument('-t', '--threshold', default=0.5, type=float,
    help='Threshold (in proportion of sentences) to score a model as understanding a word')
parser.add_argument('-r', '--rescore-function', default=None,
    help='Which function to use to rescore individual sentences')
DEFAULT_K = 3
parser.add_argument('-k', default=DEFAULT_K, help='which k to rescore for')
DEFAULT_RESULT_COL = 'result'
parser.add_argument('-c', '--result-column', default=DEFAULT_RESULT_COL)


def top_k(df, args):
    scores = [np.fromstring(x[1:-1], sep=',') for x in df.sentence_scores]
    df['new_results'] = [0 in np.argpartition(a, -args.k)[-args.k:] for a in scores]
    return aggregate_by_word_and_sentence(df, 'new_results', args)


def aggregate_by_word_and_sentence(df, args):
    return (df.groupby(['word_id', 'sentence_id']).agg('mean')[args.result_column].groupby('word_id').mean() > args.threshold).astype(int)


def model_name_from_file_name(name):
    return name[:name.find('_sentences')].replace('_', '/')


def main(args):
    input_folder_path = pathlib.Path(args.input_folder)
    input_paths = glob.glob(str((input_folder_path / '*.csv').absolute()))
    input_file_names_and_paths = [(pathlib.Path(p).name, p) for p in input_paths]

    models_to_paths = defaultdict(list)
    results = []
    models = []
    words = None

    for name, path in input_file_names_and_paths:
        models_to_paths[model_name_from_file_name(name)].append(path)

    for model_name in tqdm.tqdm(models_to_paths):
        models.append(model_name)
        df = None
        for input_file in models_to_paths[model_name]:
            if df is None:
                df = pd.read_csv(input_file)
            else:
                df = df.append(pd.read_csv(input_file))
        
        if args.rescore_function is not None:
            word_scores = globals()[args.rescore_function](df, args)
        else:
            word_scores = aggregate_by_word_and_sentence(df, args)

        results.append(list(word_scores))
        if words is None:
            words = list(df.word.unique())

    output_df = pd.DataFrame(results, columns=words)
    output_df.to_csv(input_folder_path / 'irt_data.tsv', sep='\t', index=False)
    with open(input_folder_path / 'irt_models.txt', 'w') as models_output_file:
        models_output_file.writelines([f'{model}\n' for model in models])
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

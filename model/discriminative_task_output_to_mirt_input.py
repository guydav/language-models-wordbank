import argparse
import pathlib
import glob
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--input-folder', required=True)
parser.add_argument('-t', '--threshold', default=0.5,
    help='Threshold (in proportion of sentences) to score a model as understanding a word')
parser.add_argument('-r', '--rescore-function', default=None,
    help='Which function to use to rescore individual sentences')
DEFAULT_K = 3
parser.add_argument('-k', default=DEFAULT_K, help='which k to rescore for')


def top_k(df, args):
    scores = [np.fromstring(x[1:-1], sep=',') for x in df.sentence_scores]
    df['new_results'] = [0 in np.argpartition(a, -args.k)[-args.k:] for a in scores]
    return aggregate_by_column(df, 'new_results', args)


def aggregate_by_column(df, col_name, args):
    return (df.groupby(df.word_id).agg('mean')[col_name] > args.threshold).astype(int)


def main(args):
    input_folder_path = pathlib.Path(args.input_folder)
    input_files = glob.glob(str((input_folder_path / '*.csv').absolute()))
    results = []
    models = []
    words = None
    for input_file in input_files:
        df = pd.read_csv(input_file)
        if args.rescore_function is not None:
            word_scores = globals()[args.rescore_function](df, args)
        else:
            word_scores = aggregate_by_column(df, 'result', args)
        results.append(list(word_scores))
        models.append(df.model_name[0])
        if words is None:
            words = list(df.word.unique())

    output_df = pd.DataFrame(results, columns=words)
    output_df.to_csv(input_folder_path / 'irt_data.tsv', sep='\t', index=False)
    with open(input_folder_path / 'irt_models.txt', 'w') as models_output_file:
        models_output_file.writelines([f'{model}\n' for i, model in enumerate(models)])
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

import argparse
import pathlib
import glob

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--input-folder', required=True)
parser.add_argument('-t', '--threshold', default=0.5,
    help='Threshold (in proportion of sentences) to score a model as understanding a word')
parser.add_argument('-r', '--rescore-function', default=None,
    help='Which function to use to rescore individual sentences')    


def main(args):
    input_folder_path = pathlib.Path(args.input_folder)
    input_files = glob.glob(input_folder_path / '*.csv')
    results = []
    models = []
    words = None
    for input_file in input_files:
        df = pd.read_csv(input_file)
        if args.rescore_function is not None:
            word_scores = globals()[args.rescore_function](df)
        else:
            word_scores = (df.groupby(df.word).agg('mean').result > args.threshold).astype(int)
        results.append(word_scores)
        models.append(df.model_name[0])
        if words is None:
            words = df.words.unique()

    output_df = pd.DataFrame(results, columns=words)
    output_df.to_csv(input_folder_path / 'irt_data.tsv', sep='\t')
    with open(input_folder_path / 'irt_models.txt', 'w') as models_output_file:
        models_output_file.writelines([f'{i:2d}: {model}' for i, model in enumerate(models)])
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

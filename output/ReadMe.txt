The main model to run is https://github.com/guydav/language-models-wordbank/blob/master/model/gen_eval_multiple_models_liberally_extended_GPU.py which has more models to evaluate and also uses GPU (fails without). The older CPU version of it is https://github.com/guydav/language-models-wordbank/blob/master/model/gen_eval_multiple_models.py.

It has options to decide which models to run, how many words from wordbank to test on, how many sentences to sample, number of options to generate at each mask position etc.

DEBUG variable in the beginning of the file can be toggled for printing/not printing to the terminal a lot of stuff (almost everything done in the code) as it loops through models and words. If DEBUG = 1, it's better to forward the stdout to a file (like multiple_models_run10_all_words.txt).

used_wordbank is the list of words which aren't ignored. We are ignoring multi-word words and the words in ignore_word_list (pet's name e.g.).

RUN_ID in the beginning is used for labelling output files. Fill it with a string that starts with a number followed by any additional info. E.g. 05, 07_10_words, 11_all_words. Don't use a RUN_ID already used by anybody else as it will overwrite output files.

For each model (checkpoint), this code outputs a file named "../output/"+checkpoint_name+"_RUN_" + RUN_ID+"_predictions.tsv". Each line has the format: word, childes sentence ID (line number of the sentence in childes_wordbank_cleaned_data.tsv), actual sentence from childes, start position of the word in this sentence, end position, predicted words separated by commas, their probabilities separated by commas: all the fields separated by tabs. These *_predictions.tsv files are intermediate output but more for logging purposes rather than to be used in the downstream tasks. Theoretically they could be used but we might as well run the code again.

The main output of gen_eval_multiple_models_liberally_extended_GPU.py is a file of the name: "../output/RUN_" + RUN_ID + "_scores.tsv". The list of words in the first row. Next row onwards, each row is filled with the scores (1/0) of a model for all the words in the first row. Everything is tab separated. Tt can be used as input to MIRT module through irt.R (https://github.com/guydav/language-models-wordbank/blob/master/output/irt.R). We need to add a row of all 1's and another row of all 0's as MIRT complains about words for which all the responses are either 1 or 0. We could alternatively ignore those words before running IRT. It's a choice to be made.

The secondary output of gen_eval_multiple_models_liberally_extended_GPU.py is a file of the name: "../output/RUN_" + RUN_ID + "_models.txt". It has the list of models separated by the new line character.

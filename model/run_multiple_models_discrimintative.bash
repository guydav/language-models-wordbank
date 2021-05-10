#!/usr/local/bin/bash

RemainingModels=("nyu-mll/roberta-med-small-1M-1" "nyu-mll/roberta-med-small-1M-2" "nyu-mll/roberta-med-small-1M-3" "nyu-mll/roberta-base-10M-1" "nyu-mll/roberta-base-10M-2" "nyu-mll/roberta-base-10M-3"  "nyu-mll/roberta-base-100M-1" "nyu-mll/roberta-base-100M-2" "nyu-mll/roberta-base-100M-3" "nyu-mll/roberta-base-1B-1" "nyu-mll/roberta-base-1B-2" "nyu-mll/roberta-base-1B-3" "bert-base-uncased" "distilbert-base-uncased")

for modelName in ${RemainingModels[*]}; do
    echo python run_discriminative_task.py -r 3 -s 20 -w 19 -b 256 -c "${modelName}" -o "../output/discriminative/both-datasets-40-20"
    python run_discriminative_task.py -r 3 -s 20 -w 19 -b 256 -c "${modelName}" -o "../output/discriminative/both-datasets-40-20"
done


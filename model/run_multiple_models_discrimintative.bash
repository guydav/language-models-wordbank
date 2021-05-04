#!/usr/local/bin/bash

RemainingModels=("nyu-mll/roberta-base-100M-1" "nyu-mll/roberta-base-100M-2" "nyu-mll/roberta-base-100M-3" "nyu-mll/roberta-base-1B-1" "nyu-mll/roberta-base-1B-2" "nyu-mll/roberta-base-1B-3" "bert-base-uncased" "distilbert-base-uncased")

for modelName in ${RemainingModels[*]}; do
    python run_discriminative_task.py -c "${modelName}"
done


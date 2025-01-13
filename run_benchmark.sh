#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# This is the OpenAI LLM Engine
ENGINE="gpt-3.5-turbo"
METHOD=$1
for idx in 1 2 3 4 5
do
    for dataset in "digits" "wine"
    do
        for model in "MLP_SGD" "RandomForest" "AdaBoost"
        do
            python3 exp_bayesmark/run.py --dataset $dataset --model $model --out_name $dataset-$model-$METHOD-$idx --method $METHOD --sm_mode discriminative --engine $ENGINE
        done
    done
done
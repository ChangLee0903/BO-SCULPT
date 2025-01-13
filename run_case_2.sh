#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# This is the OpenAI LLM Engine
ENGINE="gpt-3.5-turbo"
METHOD=$1

for idx in 1 2 3 4 5
do
    python3 exp_case_II/run.py --out_name $METHOD-$idx --method $METHOD --sm_mode discriminative --engine $ENGINE
done
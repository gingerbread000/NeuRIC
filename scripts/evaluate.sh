#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DEBUG_MODE="true"

MODEL_PATH=/model_weights_to_evaluate/

RUN_NAME="EVAL_$(date +%Y-%m-%d_%H-%M-%S)"
export LOG_PATH="./logs/log_$RUN_NAME.txt"

python3 -m verl.trainer.main \
    config=examples/stage3.yaml \
    worker.actor.model.model_path=${MODEL_PATH}  \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=true \
    trainer.val_only=true \
    trainer.experiment_name=${RUN_NAME} \
    data.val_files=/path/to/val_dataset.json

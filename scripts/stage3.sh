#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export DEBUG_MODE="true"


MODEL_PATH=/model_weights_to_stage2/

RUN_NAME="stage3_$(date +%Y-%m-%d_%H-%M-%S)"
export LOG_PATH="./logs/$RUN_NAME.txt"

python3 -m verl.trainer.main \
    config=examples/config_med.yaml \
    data.train_files=/path/to/reasoning_dataset.json \
    data.val_files=/path/to/val_dataset.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${RUN_NAME} \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.fsdp.torch_dtype=bf16 \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    trainer.val_only=false \


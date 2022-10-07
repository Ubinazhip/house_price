#!/bin/sh

for model in net6 net3
do
for loss_type in mae mse
  do
    for fold in 0
    do
    CUDA_VISIBLE_DEVICES=1 python3 runner.py --model $model --epochs 60 --fold $fold --batch_size 64 --scheduler_patience 12 --lr 3e-4 --weight_decay 3e-3 --loss_type $loss_type
    done
  done
done
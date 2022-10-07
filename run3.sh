#!/bin/sh

for model in net4 net2 net3
do
for loss_type in mse mae
  do
    for fold in 2 3 4
    do
    CUDA_VISIBLE_DEVICES=2 python3 runner.py --model $model --epochs 50 --fold $fold --batch_size 64 --scheduler_patience 15 --lr 3e-3 --weight_decay 3e-4 --loss_type $loss_type
    done
  done
done
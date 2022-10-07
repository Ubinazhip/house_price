#!/bin/sh

for model in net5 net2 net3 net4 net1
do
for loss_type in mae
  do
    for fold in 0 1
    do
    CUDA_VISIBLE_DEVICES=3 python3 runner.py --model $model --epochs 100 --fold $fold --batch_size 64 --scheduler_patience 20 --lr 3e-4 --weight_decay 3e-5 --loss_type $loss_type
    done
  done
done
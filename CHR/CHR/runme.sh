#!usr/bin/env bash
# select gpu devices
export CUDA_VISIBLE_DEVICES=6,7
# train
# ../data/voc/ is the path of VOCdevkit.
python -m CHR.main --batch-size 320 |& tee -a log
 

#!/bin/bash
# select gpu devices
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m CHR.main --batch-size 320 --workers 56 --data /data2/mhassan/dhs/dataset/sixray/dataset |& tee -a log

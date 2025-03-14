# !/bin/bash
if [ ! -d "./log_train" ];then
mkdir ./log_train
else
echo "exits"
fi
sbatch run_train.sbatch

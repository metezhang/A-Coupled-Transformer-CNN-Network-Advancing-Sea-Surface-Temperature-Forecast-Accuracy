# !/bin/bash
if [ ! -d "./log_predict" ];then
mkdir ./log_predict
else
echo "exits"
fi
sbatch run_predict.sbatch

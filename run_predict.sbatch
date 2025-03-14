#!/bin/bash
#SBATCH -p normal 
#SBATCH -N 1
##SBATCH --ntasks-per-node=4
##SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH -J dl_python
#SBATCH -o ./log_predict/output.%j
#SBATCH -e ./log_predict/output.%j
##SBATCH -w e2302r6n[2-4],e2302r7n[1-4],e2303r1n1,e2310r5n[1-4],e2310r6n[1-4] 
##SBATCH -w e2302r1n4,e2302r2n[1-4],e2302r3n[1-4],e2302r4n[1-4],e2302r5n[1-3],
##e2308r3n[1-4],e2308r4n[1-4],e2308r5n[1-4],e2308r6n[1-4],
##e2309r1n[1-4],e2309r2n[1-4],e2309r3n[1-4],e2309r4n[1-4],
##e2303r2n4,e2303r3n[1-4],e2303r4n[1-4],e2303r5n[1-4],e2303r6n[1-3],
##e2303r6n4,e2303r7n[1-4],e2304r1n[1-4],e2304r2n[1-4],e2304r3n[1-3],
##e2302r5n4,e2302r6n[1-4],e2302r7n[1-4],e2303r1n[1-4],e2303r2n[1-3] 
#SBATCH --exclusive


#source ~/.bashrc
module purge
module load compiler/devtoolset/7.3.1 
#module load compiler/rocm/4.0 
#module load compiler/dtk/22.10.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
#module load apps/PyTorch/1.7.mmcv/pytorch-1.7-mmcv1.3.8-rocm-4.0.1
#module load apps/PyTorch/1.10.1a0/pytorch_1.10.1-rocm_4.0.1
#module load apps/PyTorch/1.9.0a0/pytorch_1.9-rocm_4.0.1
#module load apps/PyTorch/2.1.0/pytorch-2.1.0-dtk2310
module load apps/PyTorch/1.13.1/pytorch-1.13.1-dtk2304 
module list 
#source deactivate 
#source activate tch
which python

export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0

echo $SLURM_JOB_NODELIST 
#mkdir -p ./log
mkdir -p ./hostfile
hostfile=./hostfile/$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
rm `pwd`/hostfile/hostfile-dl -f

for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)

np=$(($np*4))

nodename=$(cat $hostfile |sed -n "1p")
echo $nodename
dist_url=`echo $nodename | awk '{print $1}'`

#for single card
#echo mpirun -np 1 --allow-run-as-root --bind-to none `pwd`/single_process.sh $dist_url inception_v3 64
#mpirun -np 1 --allow-run-as-root --bind-to none `pwd`/single_process.sh $dist_url inception_v3 64

#for one node
#echo mpirun -np $np --allow-run-as-root --bind-to none `pwd`/single_process.sh $dist_url inception_v3 64
#mpirun -np $np --allow-run-as-root --bind-to none `pwd`/single_process.sh $dist_url inception_v3 64

#for multi-gpu
echo mpirun -np $np --allow-run-as-root --hostfile ./hostfile/hostfile-dl-$SLURM_JOB_ID  --bind-to none `pwd`/run_predict.sh $dist_url
mpirun -np $np --allow-run-as-root --hostfile ./hostfile/hostfile-dl-$SLURM_JOB_ID --bind-to none `pwd`/run_predict.sh $dist_url

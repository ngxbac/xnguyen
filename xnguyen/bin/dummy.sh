#!/bin/bash
set -e
set -x

module purge
module load cuda/12.1
module load gcc/10

source ~/virtualenv/miniconda3/etc/profile.d/conda.sh
conda activate llava-next

export CUDA_HOME=/usr/local/cuda-12.1/
export CXX=/opt/rh/devtoolset-10/root/usr/bin/g++
export CC=/opt/rh/devtoolset-10/root/usr/bin/gcc


if [ -z "$1" ] # normal submit, single cluster
then
  export NNODES=${SLURM_JOB_NUM_NODES}
  master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_ADDR=$master_addr
  export CURRENT_RANK=${SLURM_PROCID}
else # submit more than one cluster
  export NNODES=${1}
  master_addr=${2}
  export MASTER_ADDR=$master_addr
  export OFFSET_RANK=${3}
  export CURRENT_RANK=$((SLURM_PROCID + OFFSET_RANK))
fi

export MASTER_PORT=2010
export INIT_METHOD="tcp://${MASTER_ADDR}:${MASTER_PORT}"


torchrun --nnodes=${NNODES} --nproc-per-node=${GPUS_PER_NODE} --master-port=${MASTER_PORT} \
    --master-addr ${MASTER_ADDR} --node-rank=${CURRENT_RANK} \
    dummy_script.py


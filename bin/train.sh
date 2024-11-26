#!/bin/bash
set -e
set -x

# module purge
# module load cuda/12.1
# module load gcc/10

#source ~/virtualenv/miniconda3/etc/profile.d/conda.sh
#conda activate llava-next

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



echo "NODELIST="${SLURM_NODELIST}

echo "MASTER_ADDR="$MASTER_ADDR
echo $SLURM_PROCID

# Slurm
SUBJECT=${SUBJECT:-1}
subject=${SUBJECT}

RUN_MODE=${RUN_MODE:-"dist"}

export TOKENIZERS_PARALLELISM=1


# export CUDA_VISIBLE_DEVICES=0
#export OFFSET_RANK=${1:-0}

if [ "$RUN_MODE" = "slurm" ]; then
        RND_MASTER_PORT=$(( ( RANDOM % 10000 )  + 1000 ))
        MASTER_PORT=${MASTER_PORT:-$RND_MASTER_PORT}
        NGPUS=${NGPUS:-1}
        distributed=True
        command="torchrun --nnodes=${NNODES} --nproc-per-node=${GPUS_PER_NODE} --master-port=${MASTER_PORT} --master-addr ${MASTER_ADDR} --node-rank=${CURRENT_RANK}"
elif [ "$RUN_MODE" = "dist" ]; then
        RND_MASTER_PORT=$(( ( RANDOM % 10000 )  + 1000 ))
        MASTER_PORT=${MASTER_PORT:-$RND_MASTER_PORT}
        NGPUS=${NGPUS:-1}
        distributed=True
        command="torchrun --nproc-per-node=${NGPUS} --master-port=${MASTER_PORT}"
elif [ "$RUN_MODE" = "slurm_sbatch" ]; then
        command="srun python"
        distributed=True
else
        command="python"
        distributed=False
fi

echo "Run command ", $command

prompt_type=individual
# prompt_type=share
lr=3e-4
saveckp_freq=20
scheduler='onecycle'
batch_size=64
epochs=100
subject=1
output_dir=logs/final_1257
data_path=data/challenge_data/
# resume='logs/mindbride_individual_2345678/last.pth'
resume='None'

# Continual learning checkpoint
# cl_checkpoint='logs/cl_3468/last.pth'
cl_checkpoint='none'

PYTHONPATH=. $command \
        main.py \
        --output_dir ${output_dir} \
        --data_path ${data_path} \
        --subject 3 4 6 8 \
        --lr ${lr} \
        --batch_size ${batch_size} \
        --epochs ${epochs} \
        --scheduler ${scheduler} \
        --prompt_type ${prompt_type} \
        --distributed \
        --saveckp_freq ${saveckp_freq} \
        --num_workers 4 \
        --resume ${resume} \
        --cl_checkpoint ${cl_checkpoint} \
        --use_fp16

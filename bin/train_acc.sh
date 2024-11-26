#!/bin/bash
set -e
set -x

# module purge
# module load cuda/12.1
# module load gcc/10

#cd ${SLURM_SUBMIT_DIR}

# source ~/virtualenv/miniconda3/etc/profile.d/conda.sh
# conda activate llava-next

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


ngpus=$(nvidia-smi --list-gpus | wc -l)
if [[ ${ngpus} == 4 ]]
then
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  echo Using 4 GPUS
  nvidia-smi --list-gpus
else
  export CUDA_VISIBLE_DEVICES=0
  echo Using Single GPU
  nvidia-smi --list-gpus
fi

prompt_type=ours
# prompt_type=share
lr=3e-4
saveckp_freq=1
scheduler='onecycle'
batch_size=32
epochs=150
subject=1
output_dir=logs/test_coco_id
data_path=data/challenge_data/
resume='None'
cl_checkpoint='none'
aux_loss_factor=1.0
clip_loss_factor=0.0
l1_loss_factor=1.0
mse_loss_factor=1.0
clip_decoder='transformer'

RUN_MODE=${RUN_MODE:-"dist"}

export NCCL_DEBUG=INFO


if [ "$RUN_MODE" = "slurm" ]; then
    PYTHONPATH=. accelerate launch \
    --num_processes=$((NNODES * GPUS_PER_NODE)) \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${CURRENT_RANK} \
    --num_machines ${NNODES} \
    main.py \
    --output_dir ${output_dir} \
    --data_path ${data_path} \
    --subject 3 \
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
    --aux_loss_factor ${aux_loss_factor} \
    --clip_loss_factor ${clip_loss_factor} \
    --l1_loss_factor ${l1_loss_factor} \
    --mse_loss_factor ${mse_loss_factor} \
    --clip_decoder ${clip_decoder} \
    --use_fp16
else
    PYTHONPATH=. accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --gpu_ids 0,1,2,3 \
    --main_process_port 21693 \
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
    --aux_loss_factor ${aux_loss_factor} \
    --clip_loss_factor ${clip_loss_factor} \
    --l1_loss_factor ${l1_loss_factor} \
    --mse_loss_factor ${mse_loss_factor} \
    --clip_decoder ${clip_decoder} \
    --use_fp16
fi

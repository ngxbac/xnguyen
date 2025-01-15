#set -x

## Get input argument for training
nnodes=1                       # Number of Requested Nodes
cluster=gpu72                   # Type of nodes: gpu72, agpu72, qgpu72
script=run.sh    # Training Script
jobname=pytorch                 # Job name
num_clusters=1
host=c1901
offset=1
logdir=slurm_logs/
cpus_per_task=64

# Reset
Color_Off='\033[0m'       # Text Reset

# Regular Colors
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White

VALID_ARGS=$(getopt -o m:h:o:n:c:s:j:l:u: --long nnodes:,cpu:,cluster:,script:,jobname,host:,offset:,num_clusters:,logdir: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -n | --nnodes)
        nnodes=${2}
        echo -e "Number of Requested Nodes: " ${Green} ${nnodes} ${Color_Off}
        shift 2
        ;;
    -c | --cluster)
        cluster=${2}
        echo -e "Requested Cluster: ${Green} ${cluster}" ${Color_Off}
        shift 2
        ;;
    -s | --script)
        script=${2}
        echo -e "Execute Script: ${Green} ${script}" ${Color_Off}
        shift 2
        ;;
    -u | --cpu)
        cpus_per_task=${2}
        echo -e "CPUS Per Task: ${Green} ${cpus_per_task}" ${Color_Off}
        shift 2
        ;;
    -m | --num_clusters)
    num_clusters=${2}
      echo -e "Number of Clusters: ${Green} ${num_clusters}" ${Color_Off}
      shift 2
      ;;
    -h | --host)
    host=${2}
      echo -e "Host: ${Green} ${host}" ${Color_Off}
      shift 2
      ;;
    -o | --offset)
    offset=${2}
      echo -e "Offset: ${Green} ${offset}" ${Color_Off}
      shift 2
      ;;
    -l | --logdir)
        logdir=${2}
        echo -e "Log Dir Name: ${Green} ${logdir}" ${Color_Off}
        shift 2
        ;;
    -j | --jobname)
        jobname=${2}
        echo -e "Job Name: ${Green} ${jobname}" ${Color_Off}
        shift 2
        ;;
    --) shift;
        break
        ;;
  esac
done


echo -e  Jobname ${Green} ${jobname} ${Color_Off} requests ${Green} ${nnodes} ${Color_Off} nodes of ${Green} ${cluster} ${Color_Off} to execute ${Green} ${script} ${Color_Off}. Number of clusters ${Green} ${num_clusters} ${Color_Off}. Host ${Green} ${host} ${Color_Off}.  Log Dir ${Green} ${logdir} ${Color_Off}.  CPUS/Task ${Green} ${cpus_per_task} ${Color_Off}


## Configurate the environment

SLURM_LOG_DIR=${logdir}
mkdir -p ${SLURM_LOG_DIR}

job_cluster=${cluster}

if [[ "${cluster}" == "qgpu72" ]]; then
    NTASKS_PER_NODE=1
    GRES='gpu:4'
    GPUS_PER_NODE=4
    QOS=gpu
    TIME=3-00:00:00
elif [[ "${cluster}" == "qgpu06" ]]; then
    NTASKS_PER_NODE=1
    GRES='gpu:4'
    GPUS_PER_NODE=4
    QOS=gpu
    TIME=06:00:00

elif [[ "${cluster}" == "agpu72" ]] || [[ "${cluster}" == "gpu72" ]]; then
    NTASKS_PER_NODE=1
    GRES='gpu:1'
    GPUS_PER_NODE=1
    QOS=gpu
    TIME=3-00:00:00
elif [[ "${cluster}" == "gpu06" ]] ||  [[ "${cluster}" == "agpu06" ]]; then
    NTASKS_PER_NODE=1
    GRES='gpu:1'
    GPUS_PER_NODE=1
    QOS=gpu
    TIME=06:00:00
elif [[ "${cluster}" == "condo" ]]; then
    NTASKS_PER_NODE=1
    GRES='gpu:4'
    GPUS_PER_NODE=4
    QOS=condo
    OPT="${OPT} --constraint csce&4a100"
    TIME=3-00:00:00
elif [[ "${cluster}" == "aimrc" ]]; then
    cluster=condo
    NTASKS_PER_NODE=1
    GRES='gpu:4'
    GPUS_PER_NODE=4
    QOS=condo
    OPT="${OPT} --constraint aimrc&4a100"
    TIME=3-00:00:00
else
  echo -e  Unknown cluster: ${Red} ${cluster} ${Color_Off}
  exit -1
fi


NTASKS=$((nnodes * NTASKS_PER_NODE))
NGPUS=$((nnodes * GPUS_PER_NODE))
export GPUS_PER_NODE
CPUS_PER_TASK=${cpus_per_task}
NNODES=$((nnodes * num_clusters))
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
full_command="sbatch \
    --output ${SLURM_LOG_DIR}/log-${jobname}-${job_cluster}-${NNODES}nodes-${TOTAL_GPUS}gpus-mrank${offset}.out \
    --ntasks ${nnodes} \
    --exclusive \
    --distribution cyclic \
    --ntasks-per-node ${NTASKS_PER_NODE} \
    --job-name ${jobname} \
    --cpus-per-task ${CPUS_PER_TASK} \
    --gres ${GRES} \
    --qos ${QOS} \
    --partition ${cluster} \
    --time ${TIME} ${OPT} \
    --dependency=afterany:538724 \
    srun_buffer.sh ${script} ${NNODES} ${host} ${offset}"

echo -e "Running Command: " ${Green} ${full_command} ${Color_Off}

${full_command}



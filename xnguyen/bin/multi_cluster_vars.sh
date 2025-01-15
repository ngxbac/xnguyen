#!/bin/bash

## Get input argument for training
nclusters=2
jobname=llava-robust
script=scripts/cvpr/finetune_clip.sh
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

VALID_ARGS=$(getopt -o n:j:s:u: --long --nclusters:,jobname:,script:,cpu: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -n | --nclusters)
        nclusters=${2}
        echo -e "Number of Clusters Used: " ${Green} ${nclusters} ${Color_Off}
        shift 2
        ;;
    -s | --script)
        script=${2}
        echo -e "Execute Script: ${Green} ${script}" ${Color_Off}
        shift 2
        ;;
    -j | --jobname)
        jobname=${2}
        echo -e "Jobname Name: ${Green} ${jobname}" ${Color_Off}
        shift 2
        ;;
    -u | --cpu)
        cpus_per_task=${2}
        echo -e "CPUS PER TASK: ${Green} ${cpus_per_task}" ${Color_Off}
        shift 2
        ;;
    --) shift;
        break
        ;;
  esac
done


echo -e Jobname ${Green} ${jobname} ${Color_Off} requests ${Green} ${nclusters} ${Color_Off} clusters to execute ${Green} ${script} ${Color_Off}. CPUS/Task ${Green} ${cpus_per_task} ${Color_Off}
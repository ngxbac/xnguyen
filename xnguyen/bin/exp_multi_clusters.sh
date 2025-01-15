source multi_cluster_vars.sh

script_path=${script}
N=${nclusters} # number of clusters
NAME=${jobname}
LOG_DIR=slurm_logs/exp_${N}_clusters_${NAME}


if [[ $N == 1 ]]
then
  ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 4 -m 3 -h c1901  -j ${NAME} -s ${script_path} -c qgpu72 -o 0
  ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 4 -m 3 -h c1901  -j ${NAME} -s ${script_path} -c condo -o 4
  ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 4 -m 3 -h c1901 -j ${NAME} -s ${script_path} -c aimrc -o 8
  # ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 4 -m 2 -h c1901 -j ${NAME} -s ${script_path} -c aimrc -o 8
  # ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 2 -m 4 -h c1901 -j ${NAME} -s ${script_path} -c condo -o 4
  # ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 2 -m 4 -h c1901 -j ${NAME} -s ${script_path} -c condo -o 6
elif [[ $N == 2 ]]
then
  ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 4 -m 1 -h c2101  -j ${NAME} -s ${script_path} -c condo -o 0
  # ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 4 -m 2 -h c2101  -j ${NAME} -s ${script_path} -c aimrc -o 4
  # ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 2 -m 2 -h c2101  -j ${NAME} -s ${script_path} -c aimrc -o 4
  # ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 2 -m 2 -h c2101  -j ${NAME} -s ${script_path} -c condo -o 2
  # ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 4 -m 2 -h c2101  -j ${NAME} -s ${script_path} -c condo -o 4
  # ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 4 -m 2 -h c2101  -j ${NAME} -s ${script_path} -c qgpu72 -o 4
  # ./bin/multi_submit.sh -l ${LOG_DIR} -u ${cpus_per_task} -n 4 -m 2 -h c1901  -j ${NAME} -s ${script_path} -c aimrc -o 4
elif [[ $N == 3 ]]
then
  #./bin/multi_submit.sh -l ${LOG_DIR} -n 4 -m 3 -h c2101 -j ${NAME} -s ${script_path} -c condo -o 0
  #./bin/multi_submit.sh -l ${LOG_DIR} -n 4 -m 3 -h c2101 -j ${NAME} -s ${script_path} -c qgpu72 -o 4
  #./bin/multi_submit.sh -l ${LOG_DIR} -n 4 -m 3 -h c2101 -j ${NAME} -s ${script_path} -c aimrc -o 8
  ./bin/multi_submit.sh -l ${LOG_DIR} -n 4 -m 3 -h c1907 -j ${NAME} -s ${script_path} -c agpu72 -o 0
  ./bin/multi_submit.sh -l ${LOG_DIR} -n 4 -m 3 -h c1907 -j ${NAME} -s ${script_path} -c agpu72 -o 4
  ./bin/multi_submit.sh -l ${LOG_DIR} -n 4 -m 3 -h c1907 -j ${NAME} -s ${script_path} -c agpu72 -o 8
elif [[ $N == 6 ]]
then
    for i in `seq 0 5`;
    do
        ./bin/multi_submit.sh -l ${LOG_DIR} -n 1 -m 6 -h c2008 -j ${NAME} -s ${script_path} -c agpu72 -o $i
    done
else
  echo -e ${Red} The current script does not support submitting a job to ${N} clusters. ${Color_Off}
fi


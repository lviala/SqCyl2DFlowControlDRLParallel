#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=8gb

# Cluster Environment Setup
cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate fenicsproject


if [ $# -eq 0 ]; then
    echo "No arguments provided; -h for help"
    exit 1
fi

# if -h, output help
if [ "$1" == "-h" ]; then
    echo "A bash script to launch the parallel training automatically:"
    echo "Usage:"
    echo "qsub script_launch_parallel.sh -v first_port=firs_port,num_servers=num_servers"
    exit 0

else
    if [ $# -eq 2 ]; then
        # all good
        :
    else
        echo "Wrong number of arguments, abort; see help with -h"
        exit 1
    fi
fi

# check that all ports are free
output=$(python3 -c "from utils import bash_check_avail; bash_check_avail($2, $3)")

if [ $output == "T" ]; then
    echo "Ports available, launch..."
else
    if [ $output == "F" ]; then
        echo "Abort; some ports are not avail"
        exit 0
    else
        echo "wrong output checking ports; abort"
        exit 1
    fi
fi

# if I went so far, all ports are free: can launch!

# launch everything:

# launch servers
echo "Launching the servers. This takes a few seconds..."
let "n_sec_sleep = 10 * $2"
echo "Wait $n_sec_sleep secs for servers to start..."

python3 launch_servers.py -p $1 -n $2&

sleep $n_sec_sleep

python3 launch_parallel_training.py -p $1 -n $2

echo "Launched training!"

exit 0

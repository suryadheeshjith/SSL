#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --account=csci_ga_2572_2023sp_19

## change the partition number to use different number of GPUs
##SBATCH --partition=n1s8-v100-1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24

#SBATCH --time=16:00:00
#SBATCH --output=logs/jupyter_%j.out
#SBATCH --exclusive
#SBATCH --requeue

port=$(shuf -i 10000-65500 -n 1)

opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -N -f -R $port:localhost:$port"

for((i=1; i<=3; i++)); do
    /usr/bin/ssh $opts $USER@log-$i.nyu.cluster
done

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

Ensure your Cisco AnyConnect is running, open a terminal, run command

ssh -L $port:localhost:$port $USER@greene.hpc.nyu.edu

If you get an error saying "Host Key Verification faile" run command 
ssh-keygen -R greene.hpc.nyu.edu

Step 2:

Keep the terminal windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

Open your browser and copy paste the URL to open Jupyter. Incase there is nothing below this, wait and open this file in a while.
EOF

singularity exec --nv --overlay overlay-15GB-500K.ext3:ro\
    -B data/dataset_v2.sqsh:/dataset:image-src=/\
    -B /scratch\
    /share/apps/images/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
   /bin/bash -c "source /ext3/env.sh; jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)"

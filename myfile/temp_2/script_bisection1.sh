#!/bin/bash

# all commands that start with SBATCH contain commands that are just used by SLURM forscheduling
#################
# set a job name
#SBATCH --job-name=try
#################
# a file for job output, you can check job progress
#SBATCH --output = /scratch/wy635/output/try.out
#################
# a file for errors from the job
#SBATCH --error = /scratch/wy635/output/try.err
#################
# time you think you need; default is one hour
# in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the
# faster your job will run.
# Default is one hour, this example will run in less that 5 minutes.
#SBATCH --time=00:05:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
#SBATCH --gres gpu:4
# We are submitting to the batch partition
#SBATCH --qos = batch
#################
#number of nodes you are requesting
#SBATCH --nodes = 1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem = 8GB
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=wy635@nyu.edu

source activate py36
python3 main.py \
--path_logger '/Users/Rebecca_yao/Documents/RESEARCH/Graph/myfile/temp_2/' \
--path_gnn '/Users/Rebecca_yao/Documents/RESEARCH/Graph/myfile/temp_2/output/' \
--path_output '/Users/Rebecca_yao/Documents/RESEARCH/Graph/myfile/temp_2/output/' \
--filename_existing_gnn '' \
--generative_model 'ErdosRenyi' \
--num_examples_train 10 \
--num_examples_test 1000 \
--num_ysampling 10000\
--loss_method 'policy' \
--num_nodes 50 \
--edge_density 0.5 \
--Lambda 10 \
--LambdaIncRate 0.05 \
--batch_size 1 \
--mode 'train' \
--clip_grad_norm 40.0 \
--num_features 10 \
--num_layers 30 \
--J 3 \
--print_freq 1 \
--num_classes 2 \
--lr 0.004


#!/bin/bash
#SBATCH --job-name=try
#SBATCH --output=/scratch/wy635/output/try.out
#SBATCH --error=/scratch/wy635/output/try.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=4GB

module purge
module load python/intel/2.7.12

cd /home/wy635/Graph/temp_2/
python3 main.py \

--path_logger '/home/wy635/Graph/temp_2/' \
--path_gnn '/scratch/wy635/data-out/Graph/' \
--path_output '/scratch/wy635/data-out/Graph/' \
--filename_existing_gnn '' \
--generative_model 'ErdosRenyi' \
--num_examples_train 10 \
--num_examples_test 1000 \
--loss_method 1 \
--num_nodes 50 \
--edge_density 0.5 \
--Lambda 10 \
--LambdaIncRate 0.05 \
--batch_size 1 \
--mode 'train' \
--clip_grad_norm 40.0 \
--num_features 10 \
--num_layers 30 \
--J 3 \
--print_freq 1 \
--num_classes 2 \
--lr 0.004
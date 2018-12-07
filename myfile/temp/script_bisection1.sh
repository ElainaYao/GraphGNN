#!/bin/bash
#SBATCH --job-name=gnn
#SBATCH --output=/scratch/wy635/output/gnn_%A.out
#SBATCH --error=/scratch/wy635/output/gnn_%A.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=4GB

module purge
module load python3/intel/3.6.3
source /home/wy635/pytorch/py3.6.3/bin/activate

cd /home/wy635/Graph/GraphGNN/myfile/temp_2/
python main.py \
--path_logger '/home/wy635/Graph/GraphGNN/myfile/temp_2/' \
--path_gnn '/home/wy635/Graph/GraphGNN/myfile/temp_2/output/' \
--path_output '/home/wy635/Graph/GraphGNN/myfile/temp_2/output/' \
--filename_existing_gnn '' \
--generative_model 'RegularGraph' \
--problem 'max' \
--problem0 'Cut' \
--num_examples_train 1000 \
--num_examples_test 1000 \
--num_ysampling 10 \
--loss_method 'relaxation' \
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



python3 main.py \
--path_logger '/Users/Rebecca_yao/Documents/RESEARCH/Graph/GraphGNN/myfile/temp_2/' \
--path_gnn '/Users/Rebecca_yao/Documents/RESEARCH/Graph/GraphGNN/myfile/temp_2/output/' \
--path_output '/Users/Rebecca_yao/Documents/RESEARCH/Graph/GraphGNN/myfile/temp_2/output/' \
--filename_existing_gnn '' \
--generative_model 'ErdosRenyi' \
--num_examples_train 1000 \
--num_examples_test 1000 \
--num_ysampling 10000 \
--loss_method 'relaxation' \
--problem 'max' \
--problem0 'Cut' \
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

module purge
module load python/intel/3.7

source activate py36
python3 main.py \
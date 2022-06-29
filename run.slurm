#!/bin/bash
#SBATCH --job-name=oana_rua    # job name
#SBATCH -C v100-16g                 # type of GPU, another choice: v100-16g
##SBATCH --partition=gpu_p2         # uncomment for specifying gpu partition, gpu_p2 has 32 gb of memory, useful when we need to train with large batch sizes
#SBATCH --nodes=1                   # request one node
#SBATCH --ntasks-per-node=1         # one task per node (== GPU number here
#SBATCH --gres=gpu:1                # number of GPUs (1/4 of GPUs on a 4-GPU node)
#SBATCH --cpus-per-task=10          # CPU cores per task (1/4 of a 4-GPU node core)
##SBATCH --cpus-per-task=3          # with gpu_p2: 1/8 of the 8-GPUs node

## /!\ Caution, "multithread" in Slurm refers to hyperthreading.
#SBATCH --hint=multithread          # hyperthreading is activated
##SBATCH --qos=qos_gpu-t4           # uncomment for jobs ove 20 hours
#SBATCH --time=19:59:00             # maximum execution time requested (HH:MM:SS)
#SBATCH --output=rua%j.out    # output file
#SBATCH --error=rua%j.out     # error file (here same as the output)

DATASET_NAME='rua_nli'
MODEL_TO_USE=('camembert-large-finetuned-repnum_wl_3_classes'
              'camembert-large-finetuned-rua_wl_3_classes'
              'camembert-large-finetuned-xnli_fr_3_classes'
              'camembert-large-finetuned-repnum_wl-rua_wl_3_classes'
              'camembert-large-finetuned-xnli_fr-repnum_wl-rua_wl_3_classes'
              'camembert-large-finetuned-xnli_fr_3_classes-finetuned-repnum_wl_3_classes'
              'camembert-large-finetuned-xnli_fr_3_classes-finetuned-rua_wl_3_classes'
              'camembert-large-finetuned-xnli_fr_3_classes-finetuned-repnum_wl-rua_wl_3_classes')
MODEL_REVISION='main'



module purge
module load pytorch-gpu/py3/1.10.0   # load pytorch
python compute_threshold.py $DATASET_NAME waboucay/${MODEL_TO_USE[$1]}  $MODEL_REVISION removepast_sentencecouple_contradictionshare
echo 'removepast_sentencecouple_contradictionshare achieved'
python compute_threshold.py $DATASET_NAME waboucay/${MODEL_TO_USE[$1]} $MODEL_REVISION removepast_sentencewise_contradictionshare
echo 'removepast_sentencewise_contradictionshare achieved'
python compute_threshold.py $DATASET_NAME waboucay/${MODEL_TO_USE[$1]} $MODEL_REVISION withpast_sentencecouple_contradictionshare
echo 'withpast_sentencecouple_contradictionshare achieved'
python compute_threshold.py $DATASET_NAME waboucay/${MODEL_TO_USE[$1]} $MODEL_REVISION withpast_sentencewise_contradictionshare
echo 'All strategies achieved!'



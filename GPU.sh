#!/bin/sh 

### select a GPU queue
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J SuperHAN


### request the number of GPUs
#BSUB -gpu "num=1:mode=exclusive_process"
### specify GPU type
## BSUB -R "select[gpu16gb]"


### request the number of CPU cores (at least 4x the number of GPUs)
#BSUB -n 4
### we need to request CPU memory, too (note: this is per CPU core)
#BSUB -R "rusage[mem=8GB]"
### we want to have this on a single node
#BSUB -R "span[hosts=1]"

### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00

### -- set the email address -- 
##BSUB -u <your_email_address>
### -- send notification at start -- 
# BSUB -B 
### -- send notification at completion -- 
#BSUB -N 

### -- Specify the output and error file. %J is the job-id -- 
#BSUB -oo /work3/s204122/SuperHAN/output.out 
#BSUB -eo /work3/s204122/SuperHAN/output.err



cd /work3/s204122/SuperHAN
source .venv/bin/activate

# --max_samples 50000
# python train.py --stage fan      --data /work3/s204122/hagrid  --save_dir checkpoints
python train.py --stage sr       --data /work3/s204122/hagrid  --fan_ckpt checkpoints/fan_standalone.pt
python train.py --stage superfan --data /work3/s204122/hagrid  --sr_ckpt  checkpoints/sr/best.pt

python -m eval.visualize \
    --data /work3/s204122/hagrid \
    --ckpt checkpoints/superfan/best.pt \
    --fan_ckpt checkpoints/fan_standalone.pt \
    --n_samples 8 \
    --out eval_output/ \
    --max_samples 50000

python train.py --stage fan      --data freihand --save_dir checkpoints
python train.py --stage sr       --data freihand --fan_ckpt checkpoints/fan_standalone.pt
python train.py --stage superfan --data freihand --sr_ckpt  checkpoints/sr/best.pt

python -m eval.visualize \
    --data freihand \
    --ckpt checkpoints/superfan/best.pt \
    --fan_ckpt checkpoints/fan_standalone.pt \
    --n_samples 8 \
    --out eval_output/


python train.py --stage fan --data '/home/mp/Pictures/hagridv2_512/hagrid' --save_dir checkpoints

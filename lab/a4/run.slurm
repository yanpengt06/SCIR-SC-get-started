#!/bin/bash

#SBATCH -J nmt-3                              # 作业名为 test
#SBATCH -o slurm-%j.out                           # 屏幕上的输出文件重定向到 slurm-jobid.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 12:00:00                            # 任务运行的最长时间为 1 小时  
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1

                          
# 指定运行作业的节点是 gpu06，若不填写系统自动分配节点

source ~/.bashrc

# 设置运行环境
conda activate local_nmt


if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en --dev-src=./chr_en_data/dev.chr --dev-tgt=./chr_en_data/dev.en --vocab=vocab.json --cuda --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python3 run.py decode model.bin ./chr_en_data/test.chr ./chr_en_data/test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python3 run.py train --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en --dev-src=./chr_en_data/dev.chr --dev-tgt=./chr_en_data/dev.en --vocab=vocab.json --lr=5e-4
elif [ "$1" = "test_local" ]; then
    python3 run.py decode model.bin ./chr_en_data/test.chr ./chr_en_data/test.en outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python3 vocab.py --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi

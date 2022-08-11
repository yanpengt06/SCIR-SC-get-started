#!/bin/bash
#SBATCH -J nbdataset-test                               # 作业名为 test
#SBATCH -o log2.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 2:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1              # 单个节点使用 1 块 GPU 卡


source ~/.bashrc

# 设置运行环境
conda activate local_nmt

# 输入要执行的命令，例如 ./hello 或 python test.py 等

## pre-train
#python src/run.py pretrain vanilla wiki.txt  --writing_params_path vanilla.pretrain.params
#
## finetune
#python src/run.py finetune vanilla wiki.txt --reading_params_path vanilla.pretrain.params --writing_params_path vanilla.finetune.params \
#--finetune_corpus_path birth_places_train.tsv

# dev
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.pretrain.dev.predictions

# test
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_test_inputs.tsv \
--outputs_path vanilla.pretrain.test.predictions
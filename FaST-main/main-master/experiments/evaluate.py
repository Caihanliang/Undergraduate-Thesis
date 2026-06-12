# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import basicts


def parse_args():
    parser = ArgumentParser(description='Evaluate time series forecasting model in BasicTS framework!')
    # enter your config file path
    parser.add_argument('-cfg', '--config', default='FaST/config.py', help='training config')
    # enter your own checkpoint file path
    parser.add_argument('-ckpt', '--checkpoint', default='checkpoints/FaST/SD/SD_50_96_12/92847d81b9634a0a854859d11304d6c2/FaST_best_val_MAE.pt')
    parser.add_argument('-g', '--gpus', default='0')
    parser.add_argument('-d', '--device_type', default='gpu')
    parser.add_argument('-b', '--batch_size', default=None) # use the batch size in the config file

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    basicts.launch_evaluation(cfg=args.config, ckpt_path=args.checkpoint, device_type=args.device_type, gpus=args.gpus, batch_size=args.batch_size)
# # 快速测试模型在测试集上的性能
"""
python main-master/experiments/evaluate.py -cfg main-master/FaST/HNGS_24_8LC.py -ckpt main-master/checkpoints/FaST/HNGS_LC_50_24_8/727bd5e506f06d14ac9f52e57f203ab1/FaST_best_val_MAE.pt
 /home/user/Downloads/cai/FaST-main-24_8/FaST-main/main-master/FaST/HNGS_24_8LC.py
python main-master/experiments/evaluate.py -cfg main-master/FaST/HNGS_8_8LC.py -ckpt main-master/checkpoints/FaST/HNGS_LC_50_8_8/7bcb3a0ef45f2a47d45246b0ea0d461c/FaST_best_val_MAE.pt


cd /home/user/Downloads/cai/FaST-main-24_8/FaST-main
python main-master/experiments/evaluate.py \
  -cfg FaST/HNGS_24_8LC.py \
  -ckpt checkpoints/FaST/HNGS_LC_50_24_8_1/727bd5e506f06d14ac9f52e57f203ab1/FaST_best_val_MAE.pt


"""
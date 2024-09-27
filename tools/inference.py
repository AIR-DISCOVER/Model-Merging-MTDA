# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Modification of config and checkpoint to support legacy models
# - Add inference mode and HRDA output flag

import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.models.builder import build_train_model

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('data_config', help='test config file path')
    parser.add_argument('--show_dir', default='inference_out', help='directory where painted images will be saved')
    parser.add_argument('--model_config', default='configs/hrda/gtaHR2csHR_hrda.py', help='test config file path')
    parser.add_argument('--checkpoint', default='work_dirs/local-basic/230220_2226_gtaHR2csHR_hrda_s1_cc944/latest.pth', help='checkpoint file')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--eval_model', type=str, default=None, choices=['teacher', 'student1', 'student2'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("Model config: ", args.model_config)
    print("Checkpoint file: ", args.checkpoint)

    model_cfg = mmcv.Config.fromfile(args.model_config)

    if args.eval_model:
        model_cfg.uda.eval_model = args.eval_model

    model_cfg.model.pretrained = None
    model_cfg.data.test.test_mode = True

    dataset = build_dataset(mmcv.Config.fromfile(args.data_config).data.test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=model_cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model_cfg.model.train_cfg = None
    # model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = build_train_model(model_cfg, train_cfg=model_cfg.get('train_cfg'), test_cfg=model_cfg.get('test_cfg'))
    # breakpoint()

    # avoid custom checkpoint loading
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    efficient_test = False
  
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, True, args.show_dir, efficient_test, args.opacity)

if __name__ == '__main__':
    main()

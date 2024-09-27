
import argparse
import os
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import copy

import mmcv
import torch

from mmseg.models.builder import build_train_model
import argparse

import re

parser = argparse.ArgumentParser()
parser.add_argument("--model_1", type=str, required=True)
parser.add_argument("--model_2", type=str, required=True)
parser.add_argument("--model_3", type=str, default="")
parser.add_argument("--model_4", type=str, default="")
parser.add_argument("--model_num", type=int, default=2)
parser.add_argument("--merge_mode", choices=["liner_merge", "git_rebasin"], type=str, required=True)
parser.add_argument("--save_name", type=str, default="TempStorage.pth")
parser.add_argument("--weight", type=float, default=-1)
parser.add_argument("--sep_head", action="store_true")
parser.add_argument("--merge_buffer", action="store_true")
args = parser.parse_args()

SAVE_ROOT = "/data/discover-08/liwy/workspace/Model-Merging-MTDA"

if args.weight == -1:
    args.weight = 1 / args.model_num

#============ Model Load ==============#
model1_path = args.model_1
model2_path = args.model_2
model3_path = args.model_3
model4_path = args.model_4

def is_cs(path):
    return "2cs" in path
def is_idd(path):
    return "2idd" in path
def is_acdc(path):
    return "2acdc" in path
def is_darkzurich(path):
    return "2darkzurich" in path

def is_r101(path):
    return "r101" in path

def get_cfg_path(path, return_raw=False):
    if path == "":
        return ""
    CONFIGs = {
        "cs_r101": "configs/hrda/gtaHR2csHR_hrda_r101.py",
        "idd_r101": "configs/hrda/gtaHR2iddHR_hrda_r101.py",
        "acdc_r101": "configs/hrda/gtaHR2acdcHR_hrda_r101.py",
        "darkzurich_r101": "configs/hrda/gtaHR2darkzurichHR_hrda_r101.py",
        "cs_mitb5": "configs/hrda/gtaHR2csHR_hrda.py",
        "idd_mitb5": "configs/hrda/gtaHR2iddHR_hrda.py",
        "acdc_mitb5": "configs/hrda/gtaHR2acdcHR_hrda.py",
        "darkzurich_mitb5": "configs/hrda/gtaHR2darkzurichHR_hrda.py",
    }
    
    suffix = "r101" if is_r101(path) else "mitb5"
    prefix = "cs" if is_cs(path) else "idd" if is_idd(path) else "acdc" if is_acdc(path) else "darkzurich" if is_darkzurich(path) else ""
    if prefix == "":
        raise ValueError("Unknown prefix")
    if return_raw:
        return prefix+"_"+suffix
    else:
        return CONFIGs[prefix+"_"+suffix]

def lerp(lam, t1, t2):
  t3 = copy.deepcopy(t2)
  for p in t1: 
    t3[p] = (1 - lam) * t1[p] + lam * t2[p]
  return t3


cfg1_path = get_cfg_path(model1_path)
cfg2_path = get_cfg_path(model2_path)
cfg3_path = get_cfg_path(model3_path)
cfg4_path = get_cfg_path(model4_path)

cfg1 = mmcv.Config.fromfile(cfg1_path)
cfg2 = mmcv.Config.fromfile(cfg2_path)
cfg1.model.train_cfg = None
cfg2.model.train_cfg = None


if cfg3_path != "":
    cfg3 = mmcv.Config.fromfile(cfg3_path)
    cfg3.model.train_cfg = None
if cfg4_path != "":
    cfg4 = mmcv.Config.fromfile(cfg4_path)
    cfg4.model.train_cfg = None

model1 = build_train_model(cfg1, train_cfg=cfg1.get('train_cfg'), test_cfg=cfg1.get('test_cfg'))
model2 = build_train_model(cfg2, train_cfg=cfg2.get('train_cfg'), test_cfg=cfg2.get('test_cfg'))
if cfg3_path != "":
    model3 = build_train_model(cfg3, train_cfg=cfg3.get('train_cfg'), test_cfg=cfg3.get('test_cfg'))
if cfg4_path != "":
    model4 = build_train_model(cfg4, train_cfg=cfg4.get('train_cfg'), test_cfg=cfg4.get('test_cfg'))

target_model = build_train_model(cfg1, train_cfg=cfg1.get('train_cfg'), test_cfg=cfg1.get('test_cfg'))

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]

model1.CLASSES = CLASSES
model1.PALETTE = PALETTE
model2.CLASSES = CLASSES
model2.PALETTE = PALETTE
target_model.CLASSES = CLASSES
target_model.PALETTE = PALETTE
if cfg3_path != "":
    model3.CLASSES = CLASSES
    model3.PALETTE = PALETTE
if cfg4_path != "":
    model4.CLASSES = CLASSES
    model4.PALETTE = PALETTE

model1_checkpoint = torch.load(model1_path)
model1.load_state_dict(model1_checkpoint['state_dict'])
model2_checkpoint = torch.load(model2_path)
model2.load_state_dict(model2_checkpoint['state_dict'])
if cfg3_path != "":
    model3_checkpoint = torch.load(model3_path)
    model3.load_state_dict(model3_checkpoint['state_dict'])
if cfg4_path != "":
    model4_checkpoint = torch.load(model4_path)
    model4.load_state_dict(model4_checkpoint['state_dict'])


full_state_dict1 = model1.model.backbone.state_dict()
filtered_state_dict1 = {}
for name, param in full_state_dict1.items():
    if "bias" in name or "weight" in name:
        filtered_state_dict1[name] = param

full_state_dict2 = model2.model.backbone.state_dict()
filtered_state_dict2 = {}
for name, param in full_state_dict2.items():
    if "bias" in name or "weight" in name:
        filtered_state_dict2[name] = param


#============ Model Merge ==============#
if(args.merge_mode =="git_rebasin"):
    ## Method 1 : git Re-basin

    from gitrebasin import weight_matching, apply_permutation, resnet101_permutation_spec
    permutation_spec = resnet101_permutation_spec()
    final_permutation = weight_matching(permutation_spec, filtered_state_dict1, filtered_state_dict2, max_iter=100)
    updated_params = apply_permutation(permutation_spec, final_permutation, filtered_state_dict2)

    model2.model.backbone.load_state_dict(updated_params,strict=False)

    # count the parameter of target_model.model.backbone and target_model.model.decode_head
    print("target_model.model.backbone")
    print("num of parameters: ", sum(p.numel() for p in target_model.model.backbone.parameters()))
    
    print("target_model.model.decode_head")
    print("num of parameters: ", sum(p.numel() for p in target_model.model.decode_head.parameters()))
    
    model1_dict = copy.deepcopy(model1.state_dict())
    model2_dict = copy.deepcopy(model2.state_dict())
    naive_p = lerp(0.5, model1_dict, model2_dict)
    model2.load_state_dict(naive_p)
    torch.save({'state_dict': model2.state_dict()}, os.path.join(SAVE_ROOT, args.save_name))


elif(args.merge_mode =="liner_merge"):
    pass

else:
    raise NotImplementedError

# Linear merge weights
    
original_state_dict_keys = set(target_model.model.state_dict().keys())
changed_state_dict_keys = set()
buffer_state_dict_keys = set()

target_model.load_state_dict(model1.state_dict(), strict=False)

weight = args.weight
if args.model_num == 2:
    for target_param, (model1_param_name, model1_param), (model2_param_name, model2_param) in zip(target_model.model.parameters(), 
                                                        model1.model.named_parameters(), 
                                                        model2.model.named_parameters()):
        # print(target_param[0, 0, 0, 0].item(), model1_param[0, 0, 0, 0].item(), model2_param[0, 0, 0, 0].item())
        if not target_param.data.shape:
            target_param.data = weight * model1_param.data + (1 - weight) * model2_param.data
        else:
            target_param.data[:] = weight * model1_param[:].data[:] + (1 - weight) * model2_param[:].data[:]
        changed_state_dict_keys.add(model1_param_name)
if args.model_num == 3:
    for target_param, (model1_param_name, model1_param), (model2_param_name, model2_param), (model3_param_name, model3_param) in zip(target_model.model.parameters(), 
                                                        model1.model.named_parameters(), 
                                                        model2.model.named_parameters(),
                                                        model3.model.named_parameters()):
        # print(target_param[0, 0, 0, 0].item(), model1_param[0, 0, 0, 0].item(), model2_param[0, 0, 0, 0].item())
        if not target_param.data.shape:
            assert weight - 1 / 3 < 1e-5
            target_param.data = weight * model1_param.data + weight * model2_param.data + weight * model3_param.data
        else:
            target_param.data[:] = weight * model1_param[:].data[:] + weight * model2_param[:].data[:] + weight * model3_param[:].data[:]
        changed_state_dict_keys.add(model1_param_name)
if args.model_num == 4:
    for target_param, (model1_param_name, model1_param), (model2_param_name, model2_param), (model3_param_name, model3_param), (model4_param_name, model4_param) in zip(target_model.model.parameters(), 
                                                        model1.model.named_parameters(), 
                                                        model2.model.named_parameters(),
                                                        model3.model.named_parameters(),
                                                        model4.model.named_parameters()):
        # print(target_param[0, 0, 0, 0].item(), model1_param[0, 0, 0, 0].item(), model2_param[0, 0, 0, 0].item())
        if not target_param.data.shape:
            assert weight - 1 / 4 < 1e-5
            target_param.data = weight * model1_param.data + weight * model2_param.data + weight * model3_param.data + weight * model4_param.data
        else:
            target_param.data[:] = weight * model1_param[:].data[:] + weight * model2_param[:].data[:] + weight * model3_param[:].data[:] + weight * model4_param[:].data[:]
        changed_state_dict_keys.add(model1_param_name)


if args.merge_buffer:
    if args.model_num == 2:
        for target_buffer, (model1_buffer_name, model1_buffer), (model2_buffer_name, model2_buffer) in zip(
                                                                        target_model.get_model().buffers(), 
                                                                        model1.get_model().named_buffers(),
                                                                        model2.get_model().named_buffers()):
            
            current_buffer_key_num_batches_tracked = '.'.join(model1_buffer_name.split(".")[:-1] + ["num_batches_tracked"])
            current_buffer_key_running_mean = '.'.join(model1_buffer_name.split(".")[:-1] + ["running_mean"])
            current_buffer_key_running_var = '.'.join(model1_buffer_name.split(".")[:-1] + ["running_var"])
            
            # retrieve the buffer from models
            num_batches_tracked_A = model1.model.state_dict()[current_buffer_key_num_batches_tracked]
            running_mean_A = model1.model.state_dict()[current_buffer_key_running_mean]
            running_var_A = model1.model.state_dict()[current_buffer_key_running_var]
            num_batches_tracked_B = model2.model.state_dict()[current_buffer_key_num_batches_tracked]
            running_mean_B = model2.model.state_dict()[current_buffer_key_running_mean]
            running_var_B = model2.model.state_dict()[current_buffer_key_running_var]
            
            # num_batches_tracked_A = weight
            # num_batches_tracked_B = 1 - weight
            
            total_batches_tracked = num_batches_tracked_A + num_batches_tracked_B
            running_mean = num_batches_tracked_A / total_batches_tracked * running_mean_A + \
                            num_batches_tracked_B / total_batches_tracked * running_mean_B
            running_var = num_batches_tracked_A / total_batches_tracked * (running_var_A + (running_mean_A - running_mean)**2) + \
                            num_batches_tracked_B / total_batches_tracked * (running_var_B + (running_mean_B - running_mean)**2)
            
            # assign the buffer to target_model
            if "num_batches_tracked" in model1_buffer_name:
                target_buffer.data = total_batches_tracked
                # target_buffer.data = torch.tensor(total_batches_tracked)
                pass
            if "running_mean" in model1_buffer_name:
                target_buffer.data = running_mean
            if "running_var" in model1_buffer_name:
                target_buffer.data = running_var

            changed_state_dict_keys.add(model1_buffer_name)
            buffer_state_dict_keys.add(model1_buffer_name)
    elif args.model_num == 3:
        for target_buffer, (model1_buffer_name, model1_buffer), (model2_buffer_name, model2_buffer), (model3_buffer_name, model3_buffer) in zip( 
                                                                        target_model.get_model().buffers(), 
                                                                        model1.get_model().named_buffers(),
                                                                        model2.get_model().named_buffers(),
                                                                        model3.get_model().named_buffers()):
            
            current_buffer_key_num_batches_tracked = '.'.join(model1_buffer_name.split(".")[:-1] + ["num_batches_tracked"])
            current_buffer_key_running_mean = '.'.join(model1_buffer_name.split(".")[:-1] + ["running_mean"])
            current_buffer_key_running_var = '.'.join(model1_buffer_name.split(".")[:-1] + ["running_var"])
            
            # retrieve the buffer from models
            num_batches_tracked_A = model1.model.state_dict()[current_buffer_key_num_batches_tracked]
            running_mean_A = model1.model.state_dict()[current_buffer_key_running_mean]
            running_var_A = model1.model.state_dict()[current_buffer_key_running_var]
            num_batches_tracked_B = model2.model.state_dict()[current_buffer_key_num_batches_tracked]
            running_mean_B = model2.model.state_dict()[current_buffer_key_running_mean]
            running_var_B = model2.model.state_dict()[current_buffer_key_running_var]
            num_batches_tracked_C = model3.model.state_dict()[current_buffer_key_num_batches_tracked]
            running_mean_C = model3.model.state_dict()[current_buffer_key_running_mean]
            running_var_C = model3.model.state_dict()[current_buffer_key_running_var]
            
            # num_batches_tracked_A = weight
            # num_batches_tracked_B = 1 - weight
            
            total_batches_tracked = num_batches_tracked_A + num_batches_tracked_B + num_batches_tracked_C
            running_mean = num_batches_tracked_A / total_batches_tracked * running_mean_A + \
                            num_batches_tracked_B / total_batches_tracked * running_mean_B + \
                            num_batches_tracked_C / total_batches_tracked * running_mean_C
            running_var = num_batches_tracked_A / total_batches_tracked * (running_var_A + (running_mean_A - running_mean)**2) + \
                            num_batches_tracked_B / total_batches_tracked * (running_var_B + (running_mean_B - running_mean)**2) + \
                                num_batches_tracked_C / total_batches_tracked * (running_var_C + (running_mean_C - running_mean)**2)
            # assign the buffer to target_model
            if "num_batches_tracked" in model1_buffer_name:
                target_buffer.data = total_batches_tracked
                # target_buffer.data = torch.tensor(total_batches_tracked)
                pass
            if "running_mean" in model1_buffer_name:
                target_buffer.data = running_mean
            if "running_var" in model1_buffer_name:
                target_buffer.data = running_var

            changed_state_dict_keys.add(model1_buffer_name)
            buffer_state_dict_keys.add(model1_buffer_name)
    elif args.model_num == 4:
        for target_buffer, (model1_buffer_name, model1_buffer), (model2_buffer_name, model2_buffer), (model3_buffer_name, model3_buffer), (model4_buffer_name, model4_buffer) in zip( 
                                                                        target_model.get_model().buffers(), 
                                                                        model1.get_model().named_buffers(),
                                                                        model2.get_model().named_buffers(),
                                                                        model3.get_model().named_buffers(),
                                                                        model4.get_model().named_buffers()):
            
            current_buffer_key_num_batches_tracked = '.'.join(model1_buffer_name.split(".")[:-1] + ["num_batches_tracked"])
            current_buffer_key_running_mean = '.'.join(model1_buffer_name.split(".")[:-1] + ["running_mean"])
            current_buffer_key_running_var = '.'.join(model1_buffer_name.split(".")[:-1] + ["running_var"])
            
            # retrieve the buffer from models
            num_batches_tracked_A = model1.model.state_dict()[current_buffer_key_num_batches_tracked]
            running_mean_A = model1.model.state_dict()[current_buffer_key_running_mean]
            running_var_A = model1.model.state_dict()[current_buffer_key_running_var]
            num_batches_tracked_B = model2.model.state_dict()[current_buffer_key_num_batches_tracked]
            running_mean_B = model2.model.state_dict()[current_buffer_key_running_mean]
            running_var_B = model2.model.state_dict()[current_buffer_key_running_var]
            num_batches_tracked_C = model3.model.state_dict()[current_buffer_key_num_batches_tracked]
            running_mean_C = model3.model.state_dict()[current_buffer_key_running_mean]
            running_var_C = model3.model.state_dict()[current_buffer_key_running_var]
            num_batches_tracked_D = model4.model.state_dict()[current_buffer_key_num_batches_tracked]
            running_mean_D = model4.model.state_dict()[current_buffer_key_running_mean]
            running_var_D = model4.model.state_dict()[current_buffer_key_running_var]
            
            
            total_batches_tracked = num_batches_tracked_A + num_batches_tracked_B + num_batches_tracked_C + num_batches_tracked_D
            running_mean = num_batches_tracked_A / total_batches_tracked * running_mean_A + \
                            num_batches_tracked_B / total_batches_tracked * running_mean_B + \
                            num_batches_tracked_C / total_batches_tracked * running_mean_C + \
                            num_batches_tracked_D / total_batches_tracked * running_mean_D
            running_var = num_batches_tracked_A / total_batches_tracked * (running_var_A + (running_mean_A - running_mean)**2) + \
                            num_batches_tracked_B / total_batches_tracked * (running_var_B + (running_mean_B - running_mean)**2) + \
                            num_batches_tracked_C / total_batches_tracked * (running_var_C + (running_mean_C - running_mean)**2) + \
                            num_batches_tracked_D / total_batches_tracked * (running_var_D + (running_mean_D - running_mean)**2)
            # assign the buffer to target_model
            if "num_batches_tracked" in model1_buffer_name:
                target_buffer.data = total_batches_tracked
                # target_buffer.data = torch.tensor(total_batches_tracked)
                pass
            if "running_mean" in model1_buffer_name:
                target_buffer.data = running_mean
            if "running_var" in model1_buffer_name:
                target_buffer.data = running_var

            changed_state_dict_keys.add(model1_buffer_name)
            buffer_state_dict_keys.add(model1_buffer_name)

if not args.sep_head:
    # save model!
    torch.save({'state_dict': target_model.state_dict()}, os.path.join(SAVE_ROOT, args.save_name))

else:
    decode_head1 = model1.model.decode_head.state_dict()
    decode_head2 = model2.model.decode_head.state_dict()
    if args.model_num > 2:
        decode_head3 = model3.model.decode_head.state_dict()
    if args.model_num > 3:
        decode_head4 = model4.model.decode_head.state_dict()
    
    target_model.model.decode_head.load_state_dict(decode_head1, strict=False)
    torch.save({'state_dict': target_model.state_dict()}, os.path.join(SAVE_ROOT, args.save_name[:-4]+f"_head_{get_cfg_path(args.model_1, return_raw=True)}.pth"))
    target_model.model.decode_head.load_state_dict(decode_head2, strict=False)
    torch.save({'state_dict': target_model.state_dict()}, os.path.join(SAVE_ROOT, args.save_name[:-4]+f"_head_{get_cfg_path(args.model_2, return_raw=True)}.pth"))
    if args.model_num > 2:
        target_model.model.decode_head.load_state_dict(decode_head3, strict=False)
        torch.save({'state_dict': target_model.state_dict()}, os.path.join(SAVE_ROOT, args.save_name[:-4]+f"_head_{get_cfg_path(args.model_3, return_raw=True)}.pth"))
    if args.model_num > 3:
        target_model.model.decode_head.load_state_dict(decode_head4, strict=False)
        torch.save({'state_dict': target_model.state_dict()}, os.path.join(SAVE_ROOT, args.save_name[:-4]+f"_head_{get_cfg_path(args.model_4, return_raw=True)}.pth"))
    
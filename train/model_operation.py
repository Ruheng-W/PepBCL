# -*- coding: utf-8 -*-
# @Time    : 2021/5/7 16:50
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: data_loader_protBert.py

from configuration import config
import torch
import os

def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model)
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)

    return new_model

def save_model(model_dict, best_acc, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = 'ACC[{:.4f}], {}.pt'.format(best_acc, save_prefix)
    save_path_pt = os.path.join(save_dir, filename)
    print('save_path_pt',save_path_pt)
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)
    print('Save Model Over: {}, ACC: {:.4f}\n'.format(save_prefix, best_acc))

def adjust_model(model):
    # Freeze some layers
    # util_freeze.freeze_by_names(model, ['embedding', 'layers'])
    # util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge', 'layers'])
    # util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge', 'soft_attention', 'layers'])
    # util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge'])
    # util_freeze.freeze_by_names(model, ['embedding_merge'])
    # util_freeze.freeze_by_names(model, ['embedding'])

    # unfreeze some layers
    # for name, child in model.named_children():
    #     for sub_name, sub_child in child.named_children():
    #         if name == 'layers' and (sub_name == '3'):
    #             print('Encoder Is Unfreezing')
    #             for param in sub_child.parameters():
    #                 param.requires_grad = True

    print('-' * 50, 'Model.named_parameters', '-' * 50)
    for name, value in model.named_parameters():
        print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))

    # Count the total parameters
    params = [i for i in list(model.parameters()) if i.requires_grad == True]
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print('=' * 50, "Number of total parameters:" + str(k), '=' * 50)
    pass
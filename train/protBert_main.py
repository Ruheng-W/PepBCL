# -*- coding: utf-8 -*-
# @Time    : 2021/5/7 17:01
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: protBert_main.py

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from configuration import config as cf
from util import util_metric
from train.model_operation import save_model, adjust_model
from train.visualization import dimension_reduction, penultimate_feature_visulization
from model import prot_bert
from util import data_loader_protBert

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import seaborn as sns
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # euclidean_distance = F.pairwise_distance(output1, output2)
        cos_distance = D(output1, output2)
        # print("ED",euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 3))

        return loss_contrastive

def load_data(config):
    train_iter_orgin, test_iter = data_loader_protBert.load_data(config)
    # print('-' * 20, 'data construction over', '-' * 20)
    return train_iter_orgin, test_iter

def draw_figure_CV(config, fig_name):
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, e in enumerate(train_acc_record):
        train_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(train_loss_record):
        train_loss_record[i] = e.cpu().detach()

    for i, e in enumerate(valid_acc_record):
        valid_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(valid_loss_record):
        valid_loss_record[i] = e.cpu().detach()

    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Validation Acc Curve", fontsize=23)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_valid_interval, valid_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Validation Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_valid_interval, valid_loss_record)

    plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()


def draw_figure_train_test(config, fig_name):
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, e in enumerate(train_acc_record):
        train_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(train_loss_record):
        # train_loss_record[i] = e.cpu().detach()
        train_loss_record[i] = e

    for i, e in enumerate(test_acc_record):
        test_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(test_loss_record):
        # test_loss_record[i] = e.cpu().detach()
        test_loss_record[i] = e

    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Test Acc Curve", fontsize=23)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_test_interval, test_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Test Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_test_interval, test_loss_record)

    plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()


def cal_loss_dist_by_cosine(model):
    embedding = model.embedding
    loss_dist = 0

    vocab_size = embedding[0].tok_embed.weight.shape[0]
    d_model = embedding[0].tok_embed.weight.shape[1]

    Z_norm = vocab_size * (len(embedding) ** 2 - len(embedding)) / 2

    for i in range(len(embedding)):
        for j in range(len(embedding)):
            if i < j:
                cosin_similarity = torch.cosine_similarity(embedding[i].tok_embed.weight, embedding[j].tok_embed.weight)
                loss_dist -= torch.sum(cosin_similarity)
                # print('cosin_similarity.shape', cosin_similarity.shape)
    loss_dist = loss_dist / Z_norm
    return loss_dist


def get_loss(logits, label, criterion):
    loss = criterion(logits, label)
    loss = loss.float()
    # flooding method
    loss = (loss - config.b).abs() + config.b

    # multi-sense loss
    # alpha = -0.1
    # loss_dist = alpha * cal_loss_dist_by_cosine(model)
    # loss += loss_dist
    return loss

def get_val_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, config.num_class), label.view(-1))
    loss = (loss.float()).mean()
    # flooding method
    loss = (loss - config.b).abs() + config.b
    Q_sum = len(logits)
    logits = F.softmax(logits, dim=1)  # softmax归一化
    hat_sum_p0 = logits[:, 0].sum()/Q_sum  # 负类的概率和
    hat_sum_p1 = logits[:, 1].sum()/Q_sum  # 正类的概率和
    mul_hat_p0 = hat_sum_p0.mul(torch.log(hat_sum_p0))
    mul_hat_p1 = hat_sum_p1.mul(torch.log(hat_sum_p1))
    mul_p0 = logits[:, 0].mul(torch.log(logits[:, 0])).sum()/Q_sum
    mul_p1 = logits[:, 1].mul(torch.log(logits[:, 1])).sum()/Q_sum
    # sum_loss = loss+(-1)*(mul_hat_p0+mul_hat_p1) + 0.1*(mul_p0+mul_p1)
    sum_loss = loss+(mul_hat_p0+mul_hat_p1)-0.1*(mul_p0+mul_p1)
    return sum_loss

def periodic_test(test_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    test_metric, test_loss, test_repres_list, test_label_list, \
    test_roc_data, test_prc_data = model_eval(test_iter, model, criterion, config)

    print('test current performance')
    # print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
    print('[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
    # print(test_metric.numpy())
    plmt = test_metric.numpy()
    print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3], '%.5g\t' % plmt[4],
          '%.5g\t\t' % plmt[5], '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8], '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
    print('#' * 60 + 'Over' + '#' * 60)

    step_test_interval.append(sum_epoch)
    test_acc_record.append(test_metric[0])
    test_loss_record.append(test_loss)

    return test_metric, test_loss, test_repres_list, test_label_list


def periodic_valid(valid_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Validation' + '#' * 60)

    valid_metric, valid_loss, valid_repres_list, valid_label_list, \
    valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)

    print('validation current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(valid_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_valid_interval.append(sum_epoch)
    valid_acc_record.append(valid_metric[0])
    valid_loss_record.append(valid_loss)

    return valid_metric, valid_loss, valid_repres_list, valid_label_list


def train_model(train_iter, valid_iter, test_iter, model, optimizer, criterion, contras_criterion, config, iter_k):
    best_acc = 0
    best_performance = 0
    train_batch_loss = 0
    for epoch in range(1, config.epoch + 1):
        steps = 0
        train_epoch_loss = 0
        train_correct_num = 0
        train_total_num = 0
        current_batch_size = 0
        repres_list = []
        label_list = []
        label_b = []
        output_b = []
        logits_b = []
        model.train()
        random.shuffle(train_iter)
        for batch in train_iter:
            input, label = batch
            label = torch.tensor(label, dtype=torch.long).cuda()
            output = model.forward(input)
            logits = model.get_logits(input)
            # repres_list.extend(output.cpu().detach().numpy())
            # label_list.extend(label.cpu().detach().numpy())
            output = output.view(-1, output.size(-1))
            logits = logits.view(-1, logits.size(-1))
            label = label[1:-1]
            logits = logits[1:-1]
            output = output[1:-1]
            output_b.append(output)
            logits_b.append(logits)
            label_b.append(label)

            current_batch_size += 1
            if current_batch_size % config.batch_size == 0:
                output_b = torch.cat(output_b, dim=0)
                logits_b = torch.cat(logits_b, dim=0)
                label_b = torch.cat(label_b, dim=0)
                label_b = label_b.view(-1)
                logits_b = logits_b.view(-1, logits_b.size(-1))
                output_b = output_b.view(-1, output_b.size(-1))
                #contrastive loss
                label_ls = []
                # weight_ls = []
                contras_len = len(output_b) // 2
                label1 = label_b[:contras_len]
                label2 = label_b[contras_len:contras_len*2]
                for i in range(contras_len):
                    xor_label = (label1[i] ^ label2[i])
                    label_ls.append(xor_label.unsqueeze(0))
                    # if (label1[i] & label2[i]):
                    #     weight_ls.append(10*label1[i].unsqueeze(0))
                    # elif (label1[i] | label2[i]):
                    #     weight_ls.append((label1[i] & label2[i]).unsqueeze(0))
                    # else:
                    #     weight_ls.append((1-label1[i]).unsqueeze(0))
                contras_label = torch.cat(label_ls)
                # contras_weight = torch.cat(weight_ls)
                output1 = output_b[:contras_len]
                output2 = output_b[contras_len:contras_len*2]
                contras_loss = contras_criterion(output1, output2, contras_label)

                # ce_loss = get_loss(logits, label, criterion)
                ce_loss = criterion(logits_b, label_b)
                loss = ce_loss + contras_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps = steps + 1
                train_batch_loss = loss.item()
                train_epoch_loss += train_batch_loss

                # logits3 = torch.unsqueeze(logits, 0)
                # label3 = torch.unsqueeze(label, 0)
                corre = (torch.max(logits_b, 1)[1] == label_b).int()
                corrects = corre.sum()
                train_correct_num += corrects
                the_batch_size = label_b.size(0)
                train_total_num += the_batch_size
                train_acc = 100.0 * corrects / the_batch_size

                label_b = []
                output_b = []
                logits_b = []

                '''Periodic Train Log'''
                if steps % config.interval_log == 0:
                    # batch_pro_len = label.size(1)
                    # m = torch.zeros_like(label)
                    # B, seq_len = label.size()
                    # for i in range(B):
                    #     pro_len = label[i][0]
                    #     batch_pro_len += pro_len
                    #     for j in range(1, pro_len + 1):
                    #         m[i][j] = 1

                    # corre = torch.mm(corre, m.t())
                    # index = torch.arange(0, B).view(1, -1)
                    # corre = corre.gather(0, index)
                    # corre = torch.mul(corre, m)

                    sys.stdout.write(
                        '\rEpoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, steps,
                                                                                            train_batch_loss,
                                                                                            train_acc,
                                                                                            corrects,
                                                                                            the_batch_size))
                    print()

                    step_log_interval.append(steps)
                    train_acc_record.append(train_acc)
                    train_loss_record.append(train_batch_loss)

        sum_epoch = iter_k * config.epoch + epoch
        print(f"Train - Epoch[{epoch}] - loss: {train_epoch_loss/(len(train_iter)//config.batch_size)} | ACC: {(train_correct_num/train_total_num)*100:.4f}%({train_correct_num}/{train_total_num})")

        '''Periodic Validation'''
        if valid_iter and sum_epoch % config.interval_valid == 0:
            valid_metric, valid_loss, valid_repres_list, valid_label_list = periodic_valid(valid_iter,
                                                                                           model,
                                                                                           criterion,
                                                                                           config,
                                                                                           sum_epoch)
            valid_acc = valid_metric[0]
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_performance = valid_metric

        '''Periodic Test'''
        if test_iter and sum_epoch % config.interval_test == 0:
            time_test_start = time.time()

            test_metric, test_loss, test_repres_list, test_label_list = periodic_test(test_iter,
                                                                                      model,
                                                                                      criterion,
                                                                                      config,
                                                                                      sum_epoch)
            '''Periodic Save'''
            # save the model if specific conditions are met
            test_acc = test_metric[5]
            if test_acc > best_acc:
                best_acc = test_acc
                best_performance = test_metric
                # if config.save_best and best_acc > config.threshold:
                # torch.save({"best_auc": best_acc, "model": model.state_dict()}, f'{best_acc}.pl')

            # test_label_list = [x + 2 for x in test_label_list]
            repres_list.extend(test_repres_list)
            label_list.extend(test_label_list)

            '''feature dimension reduction'''
            if sum_epoch % 1 == 0 or epoch == 1:
                 dimension_reduction(repres_list, label_list, epoch)

            '''reduction feature visualization'''
            # if sum_epoch % 5 == 0 or epoch == 1 or (epoch % 2 == 0 and epoch <= 10):
            #     penultimate_feature_visulization(repres_list, label_list, epoch)
            #
            # time_test_end = time.time()
            # print('inference time:', time_test_end - time_test_start, 'seconds')

    return best_performance


def model_eval(data_iter, model, criterion, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)

    print('model_eval data_iter', len(data_iter))

    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        # random.shuffle(data_iter)
        for batch in data_iter:
            input, label = batch
            # input = input.cuda()
            label = torch.tensor(label, dtype=torch.long).cuda()
            # pssm = torch.tensor(pssm, dtype=torch.float).cuda()
            # input = torch.unsqueeze(input, 0)
            lll = label.clone()
            label = torch.unsqueeze(label, 0)
            # pssm = torch.unsqueeze(pssm, 0)
            # 修改
            # label = label.view(-1)
            logits = model.get_logits(input)
            output = model.forward(input)
            # logits = torch.unsqueeze(logits[:, :2], 0)

            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(lll.cpu().detach().numpy())

            # loss = criterion(logits.view(-1, config.num_class), label.view(-1))
            logits = logits.view(-1, logits.size(-1))
            label = label.view(-1)
            label = label[1:-1]
            logits = logits[1:-1]
            loss = criterion(logits, label)
            # loss = (loss.float()).mean()
            avg_loss += loss.item()

            logits = torch.unsqueeze(logits, 0)
            label = torch.unsqueeze(label, 0)
            pred_prob_all = F.softmax(logits, dim=2)
            # Prediction probability [batch_size, seq_len, class_num]
            pred_prob_positive = pred_prob_all[:, :, 1]
            positive = torch.empty([0], device=device)
            # Probability of predicting positive classes [batch_size, seq_len]
            pred_prob_sort = torch.max(pred_prob_all, 2)
            # The maximum probability of prediction in each sample [batch_size]
            pred_class = pred_prob_sort[1]
            p_class = torch.empty([0], device=device)
            la = torch.empty([0], device=device)
            # The location (class) of the predicted maximum probability in each sample [batch_size, seq_len]
            # batch_pro_len = 0
            # m = torch.zeros_like(label)
            # B, seq_len = label.size()
            # for i in range(B):
            # pro_len = label.size(1)
            #     batch_pro_len += pro_len
            positive = torch.cat([positive, pred_prob_positive[0][:]])
            p_class = torch.cat([p_class, pred_class[0][:]])
            la = torch.cat([la, label[0][:]])
                # for j in range(1, pro_len + 1):
                #     m[i][j] = 1

            corre = (pred_class == label).int()
            # corre = torch.mm(corre, m.t())
            # index = torch.arange(0, B).view(1, -1)
            # corre = corre.gather(0, index)
            # corre = torch.mul(corre, m)
            corrects += corre.sum()
            iter_size += label.size(1)
            label_pred = torch.cat([label_pred, p_class.float()])
            label_real = torch.cat([label_real, la.float()])
            pred_prob = torch.cat([pred_prob, positive])


    metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob)
    avg_loss /= len(data_iter)
    # accuracy = 100.0 * corrects / iter_size
    accuracy = metric[0]
    print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
                                                                  100*accuracy,
                                                                  corrects,
                                                                  iter_size))

    return metric, avg_loss, repres_list, label_list, roc_data, prc_data


# def k_fold_CV(train_iter_orgin, test_iter, config):
#     valid_performance_list = []

#     for iter_k in range(config.k_fold):
#         print('=' * 50, 'iter_k={}'.format(iter_k + 1), '=' * 50)

#         # Cross validation on training set
#         train_iter = [x for i, x in enumerate(train_iter_orgin) if i % config.k_fold != iter_k]
#         valid_iter = [x for i, x in enumerate(train_iter_orgin) if i % config.k_fold == iter_k]
#         print('----------Data Selection----------')
#         print('train_iter index', [i for i, x in enumerate(train_iter_orgin) if i % config.k_fold != iter_k])
#         print('valid_iter index', [i for i, x in enumerate(train_iter_orgin) if i % config.k_fold == iter_k])

#         print('len(train_iter_orgin)', len(train_iter_orgin))
#         print('len(train_iter)', len(train_iter))
#         print('len(valid_iter)', len(valid_iter))
#         if test_iter:
#             print('len(test_iter)', len(test_iter))
#         print('----------Data Selection Over----------')

#         if config.model_name == 'ACPred_LAF_Basic':
#             model = ACPred_LAF_Basic.BERT(config)
#         elif config.model_name == 'ACPred_LAF_MSE':
#             model = ACPred_LAF_MSE.BERT(config)
#         elif config.model_name == 'ACPred_LAF_MSC':
#             model = ACPred_LAF_MSC.BERT(config)
#         elif config.model_name == 'ACPred_LAF_MSMC':
#             model = ACPred_LAF_MSMC.BERT(config)

#         if config.cuda: model.cuda()
#         adjust_model(model)

#         optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.reg)
#         criterion = nn.CrossEntropyLoss()
#         model.train()

#         print('=' * 50 + 'Start Training' + '=' * 50)
#         valid_performance = train_model(train_iter, valid_iter, test_iter, model, optimizer, criterion, config, iter_k)
#         print('=' * 50 + 'Train Finished' + '=' * 50)

#         print('=' * 40 + 'Cross Validation iter_k={}'.format(iter_k + 1), '=' * 40)
#         valid_metric, valid_loss, valid_repres_list, valid_label_list, \
#         valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)
#         print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
#         print(valid_metric.numpy())
#         print('=' * 40 + 'Cross Validation Over' + '=' * 40)

#         valid_performance_list.append(valid_performance)

#         '''draw figure'''
#         draw_figure_CV(config, config.learn_name + '_k[{}]'.format(iter_k + 1))

#         '''reset plot data'''
#         global step_log_interval, train_acc_record, train_loss_record, \
#             step_valid_interval, valid_acc_record, valid_loss_record
#         step_log_interval = []
#         train_acc_record = []
#         train_loss_record = []
#         step_valid_interval = []
#         valid_acc_record = []
#         valid_loss_record = []

#     return model, valid_performance_list


def train_test(train_iter, test_iter, config):
    # print('=' * 50, 'train-test', '=' * 50)
    # print('len(train_iter)', len(train_iter))
    # print('len(test_iter)', len(test_iter))
    # START_TAG = "<START>"
    # STOP_TAG = "<STOP>"
    # label_alphabet = ['0', '1']
    # tag_to_ix = {}
    # for i in range(len(label_alphabet)):
    #     tag_to_ix[label_alphabet[i]] = i
    # tag_to_ix[START_TAG] = len(label_alphabet)
    # tag_to_ix[STOP_TAG] = len(label_alphabet) + 1

    # 加载
    model = prot_bert.BERT(config)
    # path = 'bert_finetuned_model.pkl'
    # save_model = torch.load(path)
    # model_dict = model.state_dict()
    # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    # print(state_dict.keys())
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)

    # model = prot_bert.BERT(config)
    if config.cuda:
        model.cuda()
        # model = torch.nn.DataParallel(model).cuda()
        # model = model.module
    # adjust_model(model)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.reg)
    # criterion = nn.CrossEntropyLoss()
    # criterion = focal_loss(alpha=0.5, gamma=1, num_classes=2)
    # criterion = WCE_loss(weight = 11)
    contras_criterion = ContrastiveLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 17])).to(config.device)  # weighted update (1:17)
    # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 17])).cuda()
    # criterion = DiceLoss(with_logits=True, alpha=0.01, square_denominator=True)
    # criterion = DiceLoss(with_logits=True, smooth=1e-4, ohem_ratio=0.0,
    #                             alpha=0.01, square_denominator=True,
    #                             reduction="mean")
    #weight = 9.03636364
    # criterion = Sampled_CE_loss(sample_time=25)
    # model.train()

    print('=' * 50 + 'Start Training' + '=' * 50)
    best_performance = train_model(train_iter, None, test_iter, model, optimizer, criterion, contras_criterion, config, 0)
    print('=' * 50 + 'Train Finished' + '=' * 50)

    print('*' * 60 + 'The Last Test' + '*' * 60)
    last_test_metric, last_test_loss, last_test_repres_list, last_test_label_list, \
    last_test_roc_data, last_test_prc_data = model_eval(test_iter, model, criterion, config)
    print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
    # print(last_test_metric.numpy())
    lmt = last_test_metric.numpy()
    print('%.5g\t\t' % lmt[0] , '%.5g\t\t' % lmt[1], '%.5g\t\t' % lmt[2], '%.5g\t\t' % lmt[3], '%.5g\t' % lmt[4], '%.5g\t\t' % lmt[5], '%.5g\t\t' % lmt[6],
          '%.5g\t\t' % lmt[7], '  %.5g\t\t' % lmt[8], '  %.5g\t\t' % lmt[9], ' %.5g\t\t' % lmt[10])
    print('*' * 60 + 'The Last Test Over' + '*' * 60)

    return model, best_performance, last_test_metric


def select_dataset():
    # ACP dataset
    # path_train_data = '../data/ACP_dataset/tsv/ACP-Mixed-100-train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP-Mixed-100-test.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP-Mixed-90-train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP-Mixed-90-test.tsv'

    # path_train_data = '/home/weileyi/wrh/work_space/train_new1.tsv'
    path_train_data = '/home/sde3/wrh/TR1154_no_cut.tsv'
    # path_train_data = '/home/sde3/wrh/TR640_no_cut.tsv'


    # path_train_data = '/home/weileyi/wrh/work_space/test.tsv'
    # path_test_data = '/home/weileyi/wrh/work_space/test_new1.tsv'
    path_test_data = '/home/sde3/wrh/TS125_no_cut.tsv'
    # path_test_data = '/home/sde3/wrh/TS639_no_cut.tsv'
    # path_test_data = '/home/sde3/wrh/CBH30.tsv'


    # path_train_data = '../data/ACP_dataset/tsv/ACP-Mixed-70-train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP-Mixed-70-test.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP-Mixed-60-train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP-Mixed-60-test.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP-Mixed-50-train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP-Mixed-50-test.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP-Mixed-40-train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP-Mixed-40-test.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/LEE_Dataset.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/Independent dataset.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/Independent dataset.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/LEE_Dataset.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP2_main_train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP2_main_test.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP2_alternate_train.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP2_alternate_test.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACPred-Fuse_ACP_Train500.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACPred-Fuse_ACP_Test2710.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP_FL_train_500.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP_FL_test_164.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP_DL_740.tsv'
    # path_test_data = '../data/ACP_dataset/tsv/ACP_DL_740.tsv'
    # path_train_data = '../data/ACP_dataset/tsv/ACP_DL_240.tsv'
    # path_test_data ='../data/ACP_dataset/tsv/ACP_DL_240.tsv'

    return path_train_data, path_test_data


def load_config():
    '''The following variables need to be actively determined for each training session:
       1.train-name: Name of the training
       2.path-config-data: The path of the model configuration. 'None' indicates that the default configuration is loaded
       3.path-train-data: The path of training set
       4.path-test-data: Path to test set

       Each training corresponds to a result folder named after train-name, which contains:
       1.report: Training report
       2.figure: Training figure
       3.config: model configuration
       4.model_save: model parameters
       5.others: other data
       '''

    '''Set the required variables in the configuration'''
    train_name = 'PepBCL'
    path_config_data = None
    path_train_data, path_test_data = select_dataset()

    '''Get configuration'''
    if path_config_data is None:
        config = cf.get_train_config()
    else:
        config = pickle.load(open(path_config_data, 'rb'))

    '''Modify default configuration'''
    # config.epoch = 50

    '''Set other variables'''
    # flooding method
    b = 0.06

    config.if_multi_scaled = False

    '''initialize result folder'''
    result_folder = '../result/' + config.learn_name
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    '''Save all variables in configuration'''
    config.train_name = train_name
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data

    config.b = b
    # config.if_multi_scaled = if_multi_scaled
    # config.model_name = model_name
    config.result_folder = result_folder

    return config


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=4)
    time_start = time.time()

    '''load configuration'''
    config = load_config()

    '''set device'''
    torch.cuda.set_device(config.device)

    '''load data'''
    train_iter, test_iter = load_data(config)
    print('=' * 20, 'load data over', '=' * 20)

    '''draw preparation'''
    step_log_interval = []
    train_acc_record = []
    train_loss_record = []
    step_valid_interval = []
    valid_acc_record = []
    valid_loss_record = []
    step_test_interval = []
    test_acc_record = []
    test_loss_record = []

    '''train procedure'''
    valid_performance = 0
    best_performance = 0
    last_test_metric = 0

    if config.k_fold == -1:
        # train and test
        model, best_performance, last_test_metric = train_test(train_iter, test_iter, config)
        # 保存
        # torch.save(model, 'bert_finetuned_model.pkl')
        # torch.save(model.state_dict(), 'bert_finetuned_model.pkl')
        pass

    '''draw figure'''
    draw_figure_train_test(config, config.learn_name)

    '''report result'''
    print('*=' * 50 + 'Result Report' + '*=' * 50)
    if config.k_fold != -1:
        print('valid_performance_list', valid_performance_list)
        tensor_list = [x.view(1, -1) for x in valid_performance_list]
        cat_tensor = torch.cat(tensor_list, dim=0)
        metric_mean = torch.mean(cat_tensor, dim=0)

        print('valid mean performance')
        # print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        # print('\t{}'.format(metric_mean.numpy()))
        print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
        # print(last_test_metric.numpy())
        lmt = metric_mean.numpy()
        print('%.5g\t\t' % lmt[0], '%.5g\t\t' % lmt[1], '%.5g\t\t' % lmt[2], '%.5g\t\t' % lmt[3], '%.5g\t' % lmt[4],
              '%.5g\t\t' % lmt[5], '%.5g\t\t' % lmt[6],
              '%.5g\t\t' % lmt[7], '  %.5g\t\t' % lmt[8], '  %.5g\t\t' % lmt[9], ' %.5g\t\t' % lmt[10])

        print('valid_performance list')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        for tensor_metric in valid_performance_list:
            print('\t{}'.format(tensor_metric.numpy()))
    else:
        print('last test performance')
        # print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        # print('\t{}'.format(last_test_metric))
        print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
        # print(last_test_metric.numpy())
        lmt = last_test_metric.numpy()
        print('%.5g\t\t' % lmt[0], '%.5g\t\t' % lmt[1], '%.5g\t\t' % lmt[2], '%.5g\t\t' % lmt[3], '%.5g\t' % lmt[4],
              '%.5g\t\t' % lmt[5], '%.5g\t\t' % lmt[6],
              '%.5g\t\t' % lmt[7], '  %.5g\t\t' % lmt[8], '  %.5g\t\t' % lmt[9], ' %.5g\t\t' % lmt[10])
        print()
        print('best_performance')
        # print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        # print('\t{}'.format(best_performance))
        print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
        # print(last_test_metric.numpy())
        lmt = best_performance.numpy()
        print('%.5g\t\t' % lmt[0], '%.5g\t\t' % lmt[1], '%.5g\t\t' % lmt[2], '%.5g\t\t' % lmt[3], '%.5g\t' % lmt[4],
              '%.5g\t\t' % lmt[5], '%.5g\t\t' % lmt[6],
              '%.5g\t\t' % lmt[7], '  %.5g\t\t' % lmt[8], '  %.5g\t\t' % lmt[9], ' %.5g\t\t' % lmt[10])

    print('*=' * 50 + 'Report Over' + '*=' * 50)

    '''save train result'''
    # save the model if specific conditions are met
    if config.k_fold == -1:
        best_acc = best_performance[0]
        last_test_acc = last_test_metric[0]
        if last_test_acc > best_acc:
            best_acc = last_test_acc
            best_performance = last_test_metric
            if config.save_best and best_acc >= config.threshold:
                save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)

    # save the model configuration
    with open(config.result_folder + '/config.pkl', 'wb') as file:
        pickle.dump(config, file)
    print('-' * 50, 'Config Save Over', '-' * 50)

    time_end = time.time()
    print('total time cost', time_end - time_start, 'seconds')

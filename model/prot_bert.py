# -*- coding: utf-8 -*-
# @Time    : 2021/5/7 16:50
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: data_loader_protBert.py

from transformers import BertModel, BertTokenizer
import re, torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_bert(self):
    for name, child in self.bert.named_children():
        for param in child.parameters():
            param.requires_grad = False
class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()

        global max_len, d_model, device
        # max_len = config.max_len
        d_model = 1024
        device = torch.device("cuda" if config.cuda else "cpu")

        # self.tokenizer = BertTokenizer.from_pretrained('/home/weileyi/wrh/work_space/prot_bert_bfd', do_lower_case=False)
        # self.bert = BertModel.from_pretrained("/home/weileyi/wrh/work_space/prot_bert_bfd")
        self.tokenizer = BertTokenizer.from_pretrained('/home/sde3/wrh/prot_bert_bfd', do_lower_case=False)
        self.bert = BertModel.from_pretrained("/home/sde3/wrh/prot_bert_bfd")
        # freeze_bert(self)
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=1024,
                                        out_channels=1024,
                                        kernel_size=13,
                                           stride=1,
                                           padding=6),
                              nn.ReLU(),
                              # nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
                                 )
        self.conv1d = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 1024), stride=(1, 1),
                                    padding=(0, 0)),
                                  nn.ReLU())

        # self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True)

        self.q = nn.Parameter(torch.empty(d_model,))
        self.q.data.fill_(1)
        self.block1 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        self.block2 = nn.Sequential(
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def attention(self, input, q):
        # x = q*input
        # att_weights = F.softmax(x, 2)
        att_weights = F.softmax(q, 0)
        output = torch.mul(att_weights, input)
        return output
    def forward(self, input_seq):
        # self.bert.eval()
        input_seq = ' '.join(input_seq)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        encoded_input = self.tokenizer(input_seq, return_tensors='pt')
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].cuda()
        output = self.bert(**encoded_input)
        output = output[0]
        #结合CLS特征加cnn, lstm
        # CLS_embedding = output[:, 0, :].unsqueeze(1).repeat(1, output.size(1), 1)
        # representation = torch.cat([CLS_embedding, output], dim=2)  # [batch_size, pro_len, d_model+d_model]
        # representation = representation.permute(0, 2, 1)
        # representation = self.cnn(representation)
        # representation = representation.permute(0, 2, 1)
        # representation, _ = self.lstm(representation)
        #1d Conv 融合相邻节点的特征
        # output = output.permute(0, 2, 1)
        # output = self.cnn(output)
        # output = output.permute(0, 2, 1)
        #改进的1d Conv 降维
        # output = torch.unsqueeze(output, 1)
        # output = self.conv1d(output)
        # print("conv1d: ", output.size())
        # output = output.view(1, output.size(-2), -1)
        #Attention 可注意到特征中重要的维度
        # representation = self.attention(output, self.q)
        representation = output.view(-1, 1024)
        representation = self.block1(representation)
        # representation = torch.cat([representation, self.attention(representation, self.q)], dim=2)
        # representation = representation.view(representation.size(0), -1)
        return representation
    def get_logits(self, x):
        with torch.no_grad():
            output = self.forward(x)
        logits = self.block2(output)
        return logits
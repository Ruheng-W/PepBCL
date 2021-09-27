# ---encoding:utf-8---
# @Time : 2020.12.24
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : util_file.py

def toNum(l):
    l = [float(i) for i in l]
    return l
def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []
    # CLS_labels = []
    # pssm = []
    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            l = line.split('\t')
            # if len(l[2]) <= 1000:
            sequences.append(l[0])
            # print(l[0])
            label = list(l[1])
            label = [int(i) for i in label]
            labels.append(label)
            # CLS_labels.append(int(l[2]))
            # l3 = l[3].split(',')
            # l = [toNum(i.split()) for i in l3][:-1]  # seq_len * 20
            # pssm.append(l)                           # seq_num * seq_len * 20

    return sequences, labels

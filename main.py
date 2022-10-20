# jieba
import jieba
import os
import shutil
import codecs
# Hanlp
import hanlp
# LTP
import torch
from ltp import LTP

import time
import matplotlib.pyplot as plt

file_name = "wordOut.txt"
res_name = "result.txt"


def cut_HanLP():
    source = codecs.open(file_name, encoding='utf-8')
    if os.path.exists(res_name):
        os.remove(res_name)
    result = codecs.open(res_name, 'w', encoding='utf-8')
    line = source.readline()
    line = line.rstrip('\n')
    # HanLP分词
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    while line != "":
        seg_list = tok(line)
        output = ''
        for seg in seg_list:
            output += seg
            output += ' '
        result.write(output + '\r\n')
        line = source.readline()
    else:
        print('End file')
        source.close()
        result.close()


def cut_jieba_cut():
    source = codecs.open(file_name, encoding='utf-8')
    if os.path.exists(res_name):
        os.remove(res_name)
    result = codecs.open(res_name, 'w', encoding='utf-8')
    line = source.readline()
    line = line.rstrip('\n')
    while line != "":
        # jieba分词
        seg_list = jieba.cut(line)
        output = ''
        for seg in seg_list:
            output += seg
            output += ' '
        result.write(output + '\r\n')
        line = source.readline()
    else:
        print('End file')
        source.close()
        result.close()


def cut_jieba_cut_for_search():
    source = codecs.open(file_name, encoding='utf-8')
    if os.path.exists(res_name):
        os.remove(res_name)
    result = codecs.open(res_name, 'w', encoding='utf-8')
    line = source.readline()
    line = line.rstrip('\n')
    while line != "":
        # jieba分词
        seg_list = jieba.cut_for_search(line)
        output = ''
        for seg in seg_list:
            output += seg
            output += ' '
        result.write(output + '\r\n')
        line = source.readline()
    else:
        print('End file')
        source.close()
        result.close()


def cut_LTP():
    source = codecs.open(file_name, encoding='utf-8')
    if os.path.exists(res_name):
        os.remove(res_name)
    result = codecs.open(res_name, 'w', encoding='utf-8')
    line = source.readline()
    line = line.rstrip('\n')
    ltp = LTP("LTP/small")  # 默认加载 Small 模型
    # 将模型移动到 GPU 上
    if torch.cuda.is_available():
        # ltp.cuda()
        ltp.to("cuda")
    while line != "":
        seg_list = ltp.pipeline([line], tasks=["cws"])
        output = ''
        for seg in seg_list.cws[0]:
            output += seg
            output += ' '
        result.write(output + '\r\n')
        line = source.readline()
    else:
        print('End file')
        source.close()
        result.close()


if __name__ == '__main__':
    timeNum = []
    method = ["LTP", "HanLP", "jieba.cut", "jieba.cut_for_search"]
    start = time.perf_counter()
    cut_LTP()
    end = time.perf_counter()
    timeNum.append(end - start)

    # start = time.perf_counter()
    # cut_HanLP()
    # end = time.perf_counter()
    # timeNum.append(end - start)

    start = time.perf_counter()
    cut_jieba_cut()
    end = time.perf_counter()
    timeNum.append(end - start)

    start = time.perf_counter()
    cut_jieba_cut_for_search()
    end = time.perf_counter()
    timeNum.append(end - start)
    plt.bar([0, 1, 2], timeNum)
    plt.show()

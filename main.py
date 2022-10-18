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


def cut_HanLP():
    fileName = "wordOut.txt"
    resName = "result.txt"
    source = codecs.open(fileName, 'r')
    if os.path.exists(resName):
        os.remove(resName)
    result = codecs.open(resName, 'w', encoding='utf-8')
    line = source.readline()
    line = line.rstrip('\n')
    # HanLP分词
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    while line != "":
        # jieba分词
        # segList = jieba.cut_for_search(line)
        # HanLP分词
        segList = tok(line)
        output = ''
        for seg in segList:
            output += seg
            output += ' '
        result.write(output + '\r\n')
        line = source.readline()
    else:
        print('End file')
        source.close()
        result.close()


def cut_jieba():
    fileName = "wordOut.txt"
    resName = "result.txt"
    source = codecs.open(fileName, 'r', encoding='utf-8')
    if os.path.exists(resName):
        os.remove(resName)
    result = codecs.open(resName, 'w', encoding='utf-8')
    line = source.readline()
    line = line.rstrip('\n')
    while line != "":
        # jieba分词
        segList = jieba.cut_for_search(line)
        output = ''
        for seg in segList:
            output += seg
            output += ' '
        result.write(output + '\r\n')
        line = source.readline()
    else:
        print('End file')
        source.close()
        result.close()


def cut_LTP():
    fileName = "wordOut.txt"
    resName = "result.txt"
    source = codecs.open(fileName, 'r')
    if os.path.exists(resName):
        os.remove(resName)
    result = codecs.open(resName, 'w', encoding='utf-8')
    line = source.readline()
    line = line.rstrip('\n')
    ltp = LTP("LTP/small")  # 默认加载 Small 模型
    # 将模型移动到 GPU 上
    if torch.cuda.is_available():
        # ltp.cuda()
        ltp.to("cuda")
    while line != "":
        segList = ltp.pipeline([line], tasks=["cws"])
        output = ''
        for seg in segList.cws[0]:
            output += seg
            output += ' '
        result.write(output + '\r\n')
        line = source.readline()
    else:
        print('End file')
        source.close()
        result.close()


if __name__ == '__main__':
    # cut_LTP()
    # cut_HanLP()
    cut_jieba()

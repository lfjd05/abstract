"""
    中文分词模块，参考https://www.cnblogs.com/Newsteinwell/p/6034747.html
"""
# coding=utf-8
import jieba
import os


# 找到所有txt的文件名
def find_pdf_name(file_dir):
    txt_name = []
    for root, dirs, files in os.walk(file_dir):  # 根目录，子文件目录，文件名
        for file in files:
            if file.endswith('.txt'):
                txt_name.append(os.path.join(root, file))
    return txt_name


# define this function to print a list with Chinese
def PrintListChinese(list):
    for i in range(len(list)):
        print(list[i])


class Segmentation:
    """
        中文分词类
    """
    def __init__(self, file_path, output_path):
        self.file_path = file_path
        self.output_path = output_path

    def read_file(self):
        fileTrainRead = []
        for txt_name in self.file_path:
            with open(txt_name, 'r') as fileTrainRaw:
                for line in fileTrainRaw:
                    # print(line)
                    fileTrainRead.append(line)
        return fileTrainRead

    def segment(self, data):
        # segment word with jieba
        fileTrainSeg=list(jieba.cut(data, cut_all=False))
        return fileTrainSeg

    def write_file(self, data):
        # save the result
        with open(self.output_path, 'w') as fW:
            for i in range(len(data)):
                fW.write(data[i][0])
                fW.write('\n')


# txt_names = find_pdf_name('../data/')
# Seg = Segmentation(txt_names, 'corpusSegDone.txt')
# fileTrainRead = Seg.read_file()
# fileTrainSeg = Seg.segment(fileTrainRead)  # 进行分词
# print(fileTrainSeg)
# Seg.write_file(fileTrainSeg)   # 写入文件

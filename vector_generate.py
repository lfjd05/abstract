"""
    读取和生成词向量
        该文件参考网址https://github.com/Embedding/Chinese-Word-Vectors
    网盘路径E:\BaiduNetdiskDownload下面存了两个词向量文件
"""
import os
from gensim.models import KeyedVectors
import numpy as np
# import time


class Word2Vector:
    def __init__(self, word2vector_path):
        self.path = word2vector_path
        self.name = 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
        # 打开词向量
        # 遍历文件用readline耗时65s   用for line in f 耗时63.75s
        # self.embedding_dic = {}  # 词向量索引
        #
        # with open(os.path.join(self.path, 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'),
        # encoding='utf-8') as f:
        #     embedding_len, dim = f.readline().split()
        #     # n = 0
        #     start_time = time.time()
        #     for line in f:
        #         # 获取值
        #         value = line.split()
        #         # print('当前词', value[0])
        #         self.embedding_dic[value[0]] = np.array(value[1:], dtype='float32')
        # print('词向量的数目是：%d, 向量维度是：%d, 读取耗时:%.4f' % (int(embedding_len), int(dim),
        #                                            time.time() - start_time))
        self.model = KeyedVectors.load_word2vec_format(fname=os.path.join(self.path, self.name), binary=False)

    def find(self, word):
        """
            返回维度为300的词向量
        :param word: 需要查找词向量的词
        :return: 不存在默认返回none,可以修改
        """
        try:
            return self.model[word]
        except KeyError:
            # print('该词不存在', word)
            return None

    def find_similar(self, word):
        """
            返回为 [('银行卡', 0.6969836354255676)]
        :param word:
        :return:
        """
        try:
            return self.model.most_similar(positive=word, topn=1)
        except KeyError:
            # print('该词不存在', word)
            return None


# 当此.py被当做模块导入的时候，下面的代码不会运行
if __name__ == '__main__':
    print('建立词向量索引')
    dir0 = 'D:\pycharm_programme\sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    Word2Vectorclass = Word2Vector(dir0)
    print('找到的词的词向量是', Word2Vectorclass.find('下调').shape)
    print('找到最相似的词语', Word2Vectorclass.find_similar('广东'))
    # 全0为‘，’逗号
    print('全0', Word2Vectorclass.find_similar(np.zeros((1, 300)).astype('float32')))
    print('单个字的向量', Word2Vectorclass.find('吃'))

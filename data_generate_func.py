"""
    数据生成器函数
"""
from word_segmentation import Segmentation
import re
from config_para import max_len, out_len, N_feature, Batch_size
import numpy as np


punc = '.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：℃/'
# 进行分词和统计词频
Seg = Segmentation(None, None)
chars = {}  # {词：频率}


# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    vector = [0 for _ in range(n_unique)]
    for value in sequence:
        try:
            vector[value] = 1
        except IndexError:
            print('找不到这个索引', value)
        encoding.append(vector)
    return np.array(encoding)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]


def embedding_encoder(sequence, w2c_model):
    encoding = list()
    for value in sequence:
        vector = w2c_model.find(value)
        if vector is not None:   # 向量找得到才返回，找不到不返回
            encoding.append(vector)
    return np.array(encoding)     # （seq长度，300）


def str2id(s, char2id, start_end=False):
    """

    :param s:
    :param char2id:  文字id对应词表
    :param start_end:
    :return:
    """
    # 文字转整数id
    if start_end:    # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:max_len-2]]
        ids = [2] + ids + [3]
    else:    # 普通转化
        ids = [char2id.get(c, 1) for c in s[:max_len]]
    return ids


def id2str(ids, id2char):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x0):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x0])
    return [i + [0] * (ml - len(i)) for i in x0]


def data_generator(file_name_list, char2id, batch=20, w2c_model=None):
        # 数据生成器
        train_x, train_y = [], []
        while True:
            for i in file_name_list:
                with open(i, 'r', encoding='utf-8') as f:
                    line0 = f.readlines()
                    # 去除符号
                    content_sen0 = []
                    for cnt0 in range(len(line0)):
                        input_sen0 = re.sub(r"[{}]+".format(punc), "", line0[cnt0]).lstrip()
                        if cnt0 == 0:
                            train_y.append(str2id(input_sen0, start_end=True, char2id=char2id))
                        else:
                            content_sen0 += input_sen0
                    train_x.append(str2id(content_sen0, char2id))

                if len(train_x) == batch:
                    train_x = np.array(padding(train_x))
                    train_y = np.array(padding(train_y))
                    # print('数据维度', train_x.shape, train_y.shape)   # (batch, len),(batch, 长度随着标题长度变化)
                    yield [train_x, train_y], None
                    train_x, train_y = [], []


def generate_sentence(input_sen, fitted_model, features, w2v_model, char2id, id2char, one_hot=False):
    """
        得到预测序列
    :param one_hot: 是否是独热编码形式
    :param input_sen: 要预测的序列
    :param fitted_model: 拟合好的模型
    :param features: 特征数
    :param w2v_model: 词向量模型
    :return:
    """
    # 去除的符号
    punc = '.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：℃/'
    input_sen = re.sub(r"[{}]+".format(punc), "", str(input_sen)).lstrip()
    # 进行分词, one-hot的时候不分词
    if one_hot is False:
        Seg = Segmentation(None, None)
        input_sen = Seg.segment(input_sen)

        # 分词句子转换为词向量, 并且长度补到max_len
        encoder_x = embedding_encoder(input_sen, w2v_model)   # 维度 （序列len, feature）
        encoder_x = padding(encoder_x.reshape((1, encoder_x.shape[0], features)))  # 维度 （max_len, feature）
    else:
        # encoder_x = one_hot_encode(str2id(input_sen, char2id), len(char2id)+4)
        encoder_x = one_hot_encode(str2id(input_sen, char2id), len(char2id) + 4)   # 不适用one-hot
        print('输入', str2id(input_sen, char2id))
        encoder_x = padding(encoder_x.reshape((1, encoder_x.shape[0], features)))
    # 进行预测
    y = fitted_model.predict(encoder_x)
    # print('预测结果维度', y.shape)    # （1， 100， 2759）
    y = y.reshape((y.shape[1], y.shape[2]))

    # 预测的词向量转换为词语
    n_length = y.shape[0]   # 句子长度
    output_sentence = ''
    if one_hot is False:
        for i in range(n_length):
            vector = np.reshape(y[i, :], (1, features))
            # word_list = w2v_model.most_similar(positive=vector, topn=1)
            word_list = w2v_model.find_similar(vector)
            output_sentence += word_list[0][0]
    else:
        decode_y = one_hot_decode(y)
        for k in range(n_length):
            if decode_y[k] == 3:    # 停止字符返回
                break
            word = id2str(decode_y, id2char)
            output_sentence += word
    print(output_sentence)


# 对于mse损失好像不能这么搞
# def beam_search(encoder_x, topk=3, output_len=30):
#     """
#         使用beam_search方法得到预测结果
#     :param output_len: 输出预测结果的从长度，默认标题不超过30个字
#     :param encoder_x: 编码后的输入序列
#     :param topk: 取前多少结果，如果等于1就是贪心搜索
#     :return: 预测得到的词向量
#     """
#     y = np.array([[0.0]]*topk)   # 默认开始
#     scores = [0]*topk
#     for i in range(output_len):
#         proba = model.predict(encoder_x)[:, i, 3:]

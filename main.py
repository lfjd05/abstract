# from gensim.models.word2vec import Word2Vec
# from keras.models import load_model
from keras.callbacks import Callback
from word_segmentation import Segmentation
import os
import re
import json
from data_generate_func import data_generator, id2str, str2id
from config_para import max_len, out_len, N_feature, Batch_size, char_size
# from vector_generate import Word2Vector
from model_pack import AttentionSeq2Seq
from model_pack import get_dic
import numpy as np


class Evaluate(Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 0.8

    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子，显示标题质量提高的过程
        print(gen_title(s1, topk=5))
        print(gen_title(s2))
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model1.weights')


def gen_title(s, topk=3):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s, char2id)] * topk)  # 输入转id
    yid = np.array([[2]] * topk)  # 解码均以<start>开通，这里<start>的id为2
    scores = [0] * topk  # 候选答案分数
    print('输出的尺寸', model.predict([xid, yid]).shape)
    for i in range(50):  # 强制要求标题不超过50字
        proba = model.predict([xid, yid])[:, i, 3:]  # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(proba + 1e-6)  # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _yid = []  # 暂存的候选目标序列
        _scores = []  # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j] + 3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk):  # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:]  # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == 3:  # 找到<end>就返回
                return id2str(_yid[k], id2char)
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    # 如果50字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)], id2char)


# 读取所有文本得到标题和内容文件
file = []
for root, dirs, name_list in os.walk('D:\pycharm_programme\THUCNews\THUCNews\教育'):
    for name in name_list:
        if name.endswith('txt'):
            file.append(os.path.join(root, name))
print('读取的文件数', len(file))

evaluator = Evaluate()
# 加载词向量. one-hot的时候不用
# dir0 = 'D:\pycharm_programme\sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
# Word2Vectorclass = Word2Vector(dir0)

content, title = [], []
punc = '.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：℃/'
# 进行分词和统计词频
Seg = Segmentation(None, None)
chars = {}  # {词：频率}
file_num = 0
# 词典
# 0: mask
# 1: unk
# 2: start
# 3: end
if os.path.exists('seq2seq_config.json'):
    chars, id2char, char2id = json.load(open('seq2seq_config.json'))
    id2char = {int(i): j for i, j in id2char.items()}
else:
    for i in file:
        with open(i, 'r', encoding='utf-8') as f:
            line = f.readlines()
            # 去除符号
            content_sen = []
            for cnt in range(len(line)):
                input_sen = re.sub(r"[{}]+".format(punc), "", line[cnt]).lstrip()
                # input_sen = Seg.segment(input_sen)  # 分词one-hot不用分词
                if os.path.exists('seq2seq_config.json') is False:
                    for w in input_sen:
                        chars[w] = chars.get(w, 0) + 1
                # if cnt==0:
                #     title.append(input_sen)
                # else:
                #     content_sen += input_sen
            # 正文是很多行
            # content.append(content_sen)
        file_num += 1
        if file_num % 1000 == 0:
            print('已经转化的文件数', file_num)
    id2char = {i + 4: j for i, j in enumerate(chars)}   # 给特殊字符流出空位
    char2id = {j: i for i, j in id2char.items()}
    json.dump([chars, id2char, char2id], open('seq2seq_config.json', 'w'))  # 存储词频和字典
# id:char
print('字典大小', len(chars))

# 加载预先训练的词向量
embeddings_index = get_dic()
num_word = min(len(embeddings_index), len(chars)+4)
embedding_matrix = np.zeros((num_word, char_size)).astype('float32')   # 词向量300维度
for word, i in char2id.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
        embedding_matrix[i] = embedding_vector

# for i in data_generator(file, char2id, Batch_size):  # 生成数据集, 测试数据集生成用
#     # for i in data_generator(file, char2id, 20, Word2Vectorclass):    # 生成数据集, 测试数据集生成用
#     print(i[0].shape)
#     print(i[1].shape)

# 建立模型, 训练参数在config_para.py中
Seq2Seq = AttentionSeq2Seq(max_len, out_len, N_feature, N_feature, 64)
model = Seq2Seq.define_full_model(vector=embedding_matrix)  # 各个时间步的seq2seq
# 词向量应该使用对数损失
# categorical_crossentropy只有所在的那个类别为1，其它的类别都应该是0

s1 = '在今年严峻的就业形势下，一部分大学毕业生选择考研、出国深造，以此应对金融危机带来的就业压力。' \
     '在采访中，很多大学毕业生表示，他们希望选择读研或出国留学深造来规避就业压力。南开大学商学院市场' \
     '营销专业的应届大学生陈凯告诉记者：“现在我正准备考研，提高自己各方面的理论知识和实践技能，换一个' \
     '环境来增加自己的阅历。”此外，现在出国留学费用比以前便宜不少，出国留学也成了不少毕业生的选择。'

s2 = '日前，来自广东湛江的中国学生李铁强在英国文化协会举办的2009年度第七届国际学生之星大奖赛中，获得英国' \
     '东北地区2009年度国际学生之星称号。本次大奖赛是英国规模最大的提升留英国际学生影响及贡献的比赛，共有150' \
     '0多名来自118个国家的国际学生参赛，包括李铁强在内的12位来自英国不同地区的国际学生获得地区奖。'

model.fit_generator(data_generator(file, char2id, Batch_size), steps_per_epoch=62,
                    epochs=50, callbacks=[evaluator], verbose=1)

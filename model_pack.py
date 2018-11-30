# coding=utf-8
"""
    各种seq2seq模型的定义，序列的开始符号为"_"
    attention 模型定义参考
    https://github.com/uhauha2929/examples/blob/master/Hierarchical%20Attention%20Networks%20.ipynb
"""
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Lambda, Conv1D, MaxPool1D
from keras.layers import LSTM, Bidirectional
import keras.backend as K
from numpy import array
from layer_component import AttentionDecoder
from gensim.models import KeyedVectors
import os
from config_para import char_size


class SimpleSeq2Seq:
    def __init__(self, n_input, n_output, n_uints):
        """
        :param n_input:输入维度
        :param n_output: 输出维度
        :param n_uints: 神经元128或者256
        """
        self.n_input = n_input
        self.n_output = n_output
        self.n_uints = n_uints

    def define_models(self):
        encoder_inputs = Input(shape=(None, self.n_input))  # 长度待定
        encoder = LSTM(self.n_uints, return_state=True)  # 返回状态
        encoder_output, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.n_output))
        decoder_lstm = LSTM(self.n_uints, return_state=True, return_sequences=True)
        print(encoder_states, decoder_inputs)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)  # 加状态
        decoder_dense = Dense(self.n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  # 定义模型

        # 编码器接口
        encoder_model = Model(encoder_inputs, encoder_states)
        # 解码器接口
        decoder_state_input_h = Input(shape=(self.n_uints,))
        decoder_state_input_c = Input(shape=(self.n_uints,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
        return model, encoder_model, decoder_model

    def predict_seq(self, interface_en, interface_de, source, n_steps, cardinality):
        """
        :param interface_en:编码模型
        :param interface_de:解码模型
        :param source:解码的序列
        :param n_steps:每个序列步数量
        :param cardinality:每个时间步特征数
        :return:
        """
        state = interface_en.predict(source)  # 获得语义向量
        # 输入序列输入的开始
        target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # 收集预测
        output = list()
        for t in range(n_steps):
            yhat, h, c = interface_de.predict([target_seq] + state)
            output.append(yhat[0, 0, :])
            state = [h, c]
            # 更新目标序列
            target_seq = yhat
        return array(output)


class AttentionSeq2Seq:
    def __init__(self, n_input_len, n_output_len, n_input, n_output, n_uints):
        """
        :param n_input:输入维度
        :param n_output: 输出维度
        :param n_uints: 神经元128或者256
        """
        self.n_output_len = n_output_len
        self.n_input_len = n_input_len
        self.n_input = n_input  # 输入输出特征数量
        self.n_output = n_output
        self.n_uints = n_uints
        self.vector_dim = 300   # 词向量的维度

    def define_full_model(self, vector=None):
        encoder_input = Input(shape=(None,))
        y_input = Input(shape=(None,))

        # encoder元件
        mask = Lambda(lambda x0: K.cast(K.greater(K.expand_dims(x0, 2), 0), 'float32'))
        embedding = Embedding(self.n_input, char_size, weights=[vector], trainable=True)  # 强制映射到128维度，类似做成词向量
        encoder_conv = Conv1D(self.n_input, 5)
        encoder_pool = MaxPool1D(2, strides=2)
        encoder = Bidirectional(LSTM(self.n_uints//2, return_sequences=True))

        # encoder传播
        # print('输入维度', encoder_input)
        encoder_layer1 = embedding(encoder_input)
        mask_y = mask(y_input)
        # print('词向量大小', encoder_layer1)
        encoder_layer2 = encoder_conv(encoder_layer1)
        encoder_layer3 = encoder_pool(encoder_layer2)
        # print('池化后大小', encoder_layer3)
        encoder_output = encoder(encoder_layer3)
        # print('编码器输出', encoder_output)

        # decoder元件
        decoder_att = AttentionDecoder(self.n_uints, self.n_input,
                                       return_probabilities=True)  # 这里设置输出序列维度
        dense_1 = Dense(512, activation='relu')
        dense_2 = Dense(self.n_output, activation='softmax')

        # decoder传播过程
        decoder_layer1 = decoder_att(encoder_output)
        decoder_layer2 = dense_1(decoder_layer1)
        decoder_output = dense_2(decoder_layer2)

        # 损失
        # 交叉熵作为loss，但mask掉padding部分
        print('标签和预测值为', y_input, decoder_output)
        cross_entropy = K.sparse_categorical_crossentropy(y_input[:, 1:], decoder_output[:, :-1])
        loss = K.sum(cross_entropy * mask_y[:, 1:, 0]) / K.sum(mask_y[:, 1:, 0])

        model = Model([encoder_input, y_input], decoder_output)
        model.add_loss(loss)
        model.compile(optimizer='adam', metrics=['acc'])
        model.summary()
        return model

    def final_model_predict_seq(self, interface_en, interface_de, source, n_steps, cardinality):
        """
        :param interface_en:编码模型
        :param interface_de:解码模型
        :param source:解码的序列
        :param n_steps:每个序列步数量
        :param cardinality:每个时间步特征数
        :return:
        """
        state = interface_en.predict(source)  # 获得语义向量
        # 输入序列输入的开始
        target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # 收集预测
        output = list()
        for t in range(n_steps):
            yhat, h, c = interface_de.predict([target_seq] + state)
            output.append(yhat[0, 0, :])
            state = [h, c]
            # 更新目标序列
            target_seq = yhat
        return array(output)


def get_dic():
    """
        加载词向量，得到一份字典
    """
    dir0 = 'D:\pycharm_programme\sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    name = 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    model = KeyedVectors.load_word2vec_format(fname=os.path.join(dir0, name), binary=False)
    embeddings_dic = {}
    word_vectors = model.wv
    for word, vocab_obj in model.wv.vocab.items():
        embeddings_dic[word] = word_vectors[word]
    del model, word_vectors  # 删掉gensim模型释放内存
    print('Found %s word vectors.' % len(embeddings_dic))
    return embeddings_dic

# tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)
# model = model_attention_applied_before_lstm().summary()

import jieba
import os
import gensim
import numpy as np
from tqdm import tqdm
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
import logging
from scipy.sparse import csc_matrix

'''
主要处理了从词取词向量的问题，word_to_vector_sgng和word_to_vector_ppmi两个函数从预训练的语料库中获取词向量，
Word2Vector类提供了从已有源文件中训练词向量的方法
'''


def word_to_vector_sgng():
    '''
    返回一个list，第一个值是词向量的维数，第二个值是字典，其中键为词，值为词向量
    '''
    data = []
    word_to_vector = {}
    with open('sgns.zhihu.bigram-char', 'r', encoding='utf-8') as f:
        n_line, dimension = f.readline().split(' ')
        data.append(dimension)
        for i in tqdm(range(int(n_line))):
            line = f.readline().split(' ')
            word = line[0]
            vector = map(float, line[1:-1])
            word_to_vector[word] = list(vector)
    data.append(word_to_vector)
    return data


def word_to_vector_ppmi():
    '''
        返回一个list，第一个值是词向量的维数，第二个值是字典，其中键为词，值为以稀疏值表示的词向量字典，键为第i个维度，
        值为这个维度上的值，其余未表示出来的维度为0
    '''
    data = []
    word_to_vector = {}
    with open('ppmi.zhihu.bigram-char', 'r', encoding='utf-8') as f:
        n_line, dimension = f.readline().split(' ')
        data.append(dimension)
        for i in tqdm(range(int(n_line))):
            line = f.readline().split(' ')
            word = line[0]
            vector = {}
            for value in line[1:-1]:
                index, num = value.split(':')
                vector[int(index)] = float(num)
            word_to_vector[word] = vector
    data.append(word_to_vector)
    return data


class Word2Vector():
    '''
    先运行segment_sentence对源文件进行分词，然后运行fit训练模型，最后用predict获取词向量，或者直接获得所有词向量
    '''
    def __init__(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.model = None

    def segment_sentence(self, source_path, target_path, dic_path=None):
        """
        分词，source_path是源文件地址，target_path是输出文件地址，dic_path是jieba分词自定义词典地址
        """
        if dic_path:
            jieba.load_userdict(dic_path)
        sentences = []
        with open(source_path, 'r') as sf:
            line = sf.readline()[:-1]  # 去掉换行符
            while line:
                sentence = list(jieba.cut(line))
                sentences.append(sentence)
                line = sf.readline()
        with open(target_path, 'w') as tf:
            for sentence in sentences:
                tf.write(' '.join(sentence))

    def fit(self, input_dir):
        '''
        input_dir是所有词文件存放的目录
        '''
        # embedding size:256 共现窗口大小:10 去除出现次数5以下的词,多线程运行,迭代10次
        model = Word2Vec(PathLineSentences(input_dir), size=256, window=10, min_count=5,
                         workers=multiprocessing.cpu_count(), iter=10)
        self.model = model.evaluate_word_pairs

    def predict_vec(self, word):
        '''
        word: 要获取对应词向量的词
        '''
        try:
            vec = self.model.wv[word]
            return vec
        except KeyError:
            raise ValueError('No such word in this model. Update the model and try again.')
        finally:
            return

    def get_wordvec(self):
        '''
        获取所有词向量
        '''
        if self.model:
            return self.model.wv
        else:
            return None

    def get_model(self):
        '''
        gensim训练出来的模型有很多很好用的函数，可以看文档调用
        '''
        return self.model

    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self, load_path):
        self.model = Word2Vec.load(load_path)

    def update_model(self, file_path, dic_path=None):
        '''
        更新模型，file_path是更新用的源文件，dic_path是jieba分词
        '''
        if dic_path:
            jieba.load_userdict(dic_path)
        sentences = []
        with open(file_path, 'r') as sf:
            line = sf.readline()[:-1]  # 去掉换行符
            while line:
                sentence = list(jieba.cut(line))
                sentences.append(sentence)
                line = sf.readline()
        self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
'''
data_vector = word_to_vector_sgng()
n1 = np.squeeze(np.asarray(data_vector[1]['跑步']))
n2 = np.squeeze(np.asarray(data_vector[1]['健身']))
cos1 = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
ou1 = np.sqrt(np.sum(np.square(n1 - n2)))
n3 = np.squeeze(np.asarray(data_vector[1]['电影']))
n4 = np.squeeze(np.asarray(data_vector[1]['体育']))
cos2 = np.dot(n3, n4) / (np.linalg.norm(n3) * np.linalg.norm(n4))
ou2 = np.sqrt(np.sum(np.square(n3 - n4)))
print(cos1,cos2)
print(ou1,ou2)
'''
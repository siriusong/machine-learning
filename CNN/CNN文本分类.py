import numpy as np  #用于数据维度处理
import re  #用于正则化处理
import itertools
from collections import Counter
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D  #导入输入层，全连接层，Embedding，卷积层，池化层
from keras.layers import Reshape, Flatten, Dropout, Concatenate  #导入Dropout，Flatten层
from keras.callbacks import ModelCheckpoint  #导入召回
from tensorflow.keras.optimizers import Adam  #导入Adam优化器
from keras.models import Model  #用于建立模型
from sklearn.model_selection import train_test_split  #用于测试集训练集划分

def clean_str(string):
    """
    数据清洗，正则化处理
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()  #取小写字符

def load_data_and_labels():
    """
    导入数据，进行分词处理并生成标签
    返回分词结果及标签
    """
    # 导入数据
    positive_examples = list(open("./rt-polarity.pos", "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]  #分词处理
    negative_examples = list(open("./rt-polarity.neg", "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]  #分词处理
    #合并单词
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # 标签生成
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)  #连接
    return [x_text, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    填充句子至相同长度，长度由最长句子决定
    """
    #获取最长句子长度
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]  
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences   #返回填充后句子

def build_vocab(sentences):
    # 建立词库
    word_counts = Counter(itertools.chain(*sentences))
    # 建立索引
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv] 

def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences]) #转成array格式
    y = np.array(labels)  #标签转为array
    return [x, y]

def load_data():
    """
    导入并预处理数据
    """
    # 导入并预处理数据
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv] 
x, y, vocabulary, vocabulary_inv = load_data()

#划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

#获取维度
sequence_length = x.shape[1] # 56
vocabulary_size = len(vocabulary_inv) # 18765

#设置默认值
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5
epochs = 10
batch_size = 30 

inputs = Input(shape=(sequence_length,), dtype='int32')  #定义输入层
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)  #定义Embedding层
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)  #改变维度

#定义卷积层
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
#定义池化层
maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

#连接
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)#定义Flatten层

dropout = Dropout(drop)(flatten) #定义Dropout层

output = Dense(units=2, activation='softmax')(dropout) #定义全连接层

model = Model(inputs=inputs, outputs=output)  #定义网络

adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  #定义Adam优化器

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])  #模型编译

#网络训练！
print("Traning Model...")
model.fit(X_train[:100], y_train[:100], batch_size=batch_size, epochs=10, verbose=1, validation_data=(X_test[:100], y_test[:100]))
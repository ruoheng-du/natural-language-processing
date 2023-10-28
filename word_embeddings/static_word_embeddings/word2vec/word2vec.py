import pandas as pd
from gensim.models import Word2Vec
# from nltk.tokenize import word_tokenize
import jieba

def txt_cut(s):
    res = [w for w in jieba.lcut(s) if w.strip()]
    return " ".join(res)


# 从excel文件中读取数据
df = pd.read_excel('/Users/duruoheng/Desktop/text-classification/基于静态词向量5/text.xlsx', header=None)

# 假设你的文本数据在名为'text'的列中
# df[0] = df[0].tolist()

# 对文本进行分词
tokenized_text = []
for i in df[0]:
    new = txt_cut(i).split(' ')
    tokenized_text.extend(new)

# tokenized_text = [word for word in tokenized_text if word.strip()]
# print(tokenized_text)

# 初始化Word2Vec模型
model = Word2Vec(vector_size=50, window=5, min_count=1)

# 构建词汇表
model.build_vocab([tokenized_text])

# 训练模型
model.train([tokenized_text], total_examples=model.corpus_count, epochs=model.epochs)

# 创建一个空的DataFrame来保存词汇和对应的词向量
word_vector_df = pd.DataFrame()

for word in model.wv.index_to_key:
    # 将词向量转化为Series，然后添加到DataFrame中
    word_vector_series = pd.Series([word] + list(model.wv[word]))
    word_vector_df = word_vector_df.append(word_vector_series, ignore_index=True)

# 设置列名
word_vector_df.columns = ['Word'] + ['Dimension_' + str(i) for i in range(1, model.vector_size + 1)]

# 保存为.xlsx文件
word_vector_df.to_excel('/Users/duruoheng/Desktop/text-classification/基于静态词向量5/word2vec_word_vectors.xlsx', index=False, engine='openpyxl')

# get the vector for a single word
word = "宗教"

if word in model.wv.key_to_index:
    # Get the vector for the word
    word_vector = model.wv[word]

    # Print the vector
    print(word_vector)
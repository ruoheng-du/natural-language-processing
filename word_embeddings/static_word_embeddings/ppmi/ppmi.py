from collections import defaultdict
import numpy as np
import math
import jieba
import pandas as pd

# 这是我们的文本数据
# text = "I am a software engineer. I am a data scientist. I am a machine learning engineer. I am a product manager."

# 分词
# words = text.split()


def txt_cut(s):
    res = [w for w in jieba.lcut(s)]
    return " ".join(res)


# 读取数据
df = pd.read_excel('/Users/duruoheng/Desktop/text-classification/基于静态词向量5/text.xlsx', header=None)

# 对文本进行分词
words = []
for i in df[0]:
    new = txt_cut(i).split(' ')
    words.extend(new)


# print(words)
# 构建词典
word2id = {word: i for i, word in enumerate(set(words))}
id2word = {i: word for word, i in word2id.items()}


# 构建共现矩阵
co_occurrence_matrix = np.zeros((len(word2id), len(word2id)))
for i in range(len(words)):
    for j in range(max(0, i - 5), min(i + 6, len(words))):
        if i != j:
            co_occurrence_matrix[word2id[words[i]]][word2id[words[j]]] += 1

# 计算PPMI矩阵
N = np.sum(co_occurrence_matrix)
PPMI_matrix = np.zeros_like(co_occurrence_matrix)
for i in range(len(word2id)):
    for j in range(len(word2id)):
        pmi = np.log2(co_occurrence_matrix[i][j] * N / (np.sum(co_occurrence_matrix[i, :]) * np.sum(co_occurrence_matrix[:, j])) + 1e-8)
        PPMI_matrix[i][j] = max(0, pmi)

# #====================================矩阵太大时，可以加入该模块，使用奇异值SVD分解进行降维==========================
# from numpy import linalg as LA

# # 对PPMI矩阵进行SVD分解
# U, s, V = LA.svd(PPMI_matrix)

# # 选择主要的k个奇异值
# k = 2

# # 降维
# reduced_U = U[:, :k]

# # 获取降维后的"engineer"和"data"的词向量
# engineer_vector_reduced = reduced_U[word2id["engineer"]]
# data_vector_reduced = reduced_U[word2id["data"]]

# print("Reduced engineer vector: ", engineer_vector_reduced)
# print("Reduced data vector: ", data_vector_reduced)

# 获取单词的词向量
# vector = PPMI_matrix[word2id["宗教"]]
# data_vector = PPMI_matrix[word2id["data"]]

# print("宗教 vector: ", vector)
# print("Data vector: ", data_vector)

# Create a DataFrame to store the word vectors
word_vectors = pd.DataFrame(PPMI_matrix, index=id2word.values(), columns=id2word.values())

# Save the word vectors to an Excel file
word_vectors.to_excel("/Users/duruoheng/Desktop/text-classification/基于静态词向量5/ppmi_word_vectors.xlsx", index=True)


# get the vector for a single word
word = "宗教"

if word in word2id:
    # Get the index of the word
    word_index = word2id[word]

    # Access the vector from the PPMI_matrix using the index
    word_vector = PPMI_matrix[word_index]

    # Print the vector
    print(word_vector)
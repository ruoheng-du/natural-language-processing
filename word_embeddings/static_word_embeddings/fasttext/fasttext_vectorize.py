import fasttext
import pandas as pd
import jieba

# 这是我们的原始数据，一般来说，你需要用大量的文本数据进行训练
# sentences = [
#     "I am a software engineer",
#     "I am a data scientist",
#     "I am a machine learning engineer",
#     "I am a product manager"
# ]

def txt_cut(s):
    res = [w for w in jieba.lcut(s)]
    return " ".join(res)


# 读取数据
df = pd.read_excel('/Users/duruoheng/Desktop/text-classification/基于静态词向量5/text.xlsx', header=None)

# 对文本进行分词
words = []
for i in df[0]:
    new = txt_cut(i)
    words.append(new)

# words = [word for word in words if word.strip()]

# print(words)

# 将句子写入文本文件
with open('/Users/duruoheng/Desktop/text-classification/基于静态词向量5/fasttext_data.txt', 'w') as f:
    for sentence in words:
        f.write(sentence + '\n')

# 使用FastText训练模型
model = fasttext.train_unsupervised('/Users/duruoheng/Desktop/text-classification/基于静态词向量5/fasttext_data.txt', minn=3, maxn=3, minCount = 1, dim=50)

# 获取单词的词向量
word_vector = model.get_word_vector("宗教")
print(word_vector)

# Get the words in the model's vocabulary
words = model.words

# Create a DataFrame to store word vectors
word_vector_df = pd.DataFrame(columns=['Word'] + ['Dimension_' + str(i) for i in range(1, model.get_dimension() + 1)])

# Populate the DataFrame with word vectors
for word in words:
    vector = [word] + list(model.get_word_vector(word))
    word_vector_df.loc[len(word_vector_df)] = vector

# Save the word vectors to an XLSX file
word_vector_df.to_excel('/Users/duruoheng/Desktop/text-classification/基于静态词向量5/fasttext_word_vectors.xlsx', index=False)


# Save the word vectors to a text file
# with open('/Users/duruoheng/Desktop/text-classification/基于静态词向量5/fasttext_word_vectors.txt', 'w', encoding='utf-8') as f:
#     for word in words:
#         vector = model.get_word_vector(word)
#         vector_str = ' '.join(str(val) for val in vector)
#         f.write(f"{word} {vector_str}\n")

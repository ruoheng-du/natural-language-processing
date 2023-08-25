# Implementation of Hierarchical Attentional Networks for Document Classification
# http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf



import nltk
from tensorflow import keras
import sklearn
import numpy as np
import pandas as pd
from keras import backend as K
import matplotlib.pyplot as plt
nltk.download('punkt')



# Define parameters
# Maximum words per sentence
MAX_WORDS = 100
# Maximum sentences per doc
MAX_SENT = 15
# Max vocabulary size
MAX_VOCAB = 20000
# Dimension of GloVe
GLOVE_DIM = 100



# Build a HAN layer
# https://keras.io/layers/writing-your-own-keras-layers/
class han_attention_layer(keras.layers.Layer):

    def __init__(self, output_dim=GLOVE_DIM, **kwargs):
        self.output_dim = output_dim
        super(han_attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create trainable weight variables for this layer
        dim = input_shape[1]
        self.W = self.add_weight(name='W',
                                 shape=(dim, self.output_dim),
                                 initializer=keras.initializers.get("uniform"),
                                 trainable=True)

        # Trainable weight
        self.u = self.add_weight(name='output',
                                 shape=(self.output_dim, 1),
                                 initializer=keras.initializers.get("uniform"),
                                 trainable=True)

        super(han_attention_layer, self).build(input_shape)

    def get_att_weights(self, x):
        u_tw = K.tanh(K.dot(x, self.W))
        tw_stimulus = K.dot(u_tw, self.u)
        tw_stimulus = K.reshape(tw_stimulus, (-1, tw_stimulus.shape[1]))
        return K.softmax(tw_stimulus)

    def call(self, x):
        weights = self.get_att_weights(x)
        weights = K.reshape(weights, (-1, weights.shape[1], 1))
        weights = K.repeat_elements(weights, x.shape[-1], -1)
        weighted_input = keras.layers.Multiply()([x, weights])
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]
    


# Create HAN Model
class han(keras.models.Model):

    def __init__(self, max_words, max_sents, output_size, embed_matrix,
                 word_encode_dim=200, sent_encode_dim=200,
                 name="Hierarchical_Attention_Network"):
        self.max_words = max_words
        self.max_sents = max_sents
        self.output_size = output_size
        self.embed_matrix = embed_matrix
        self.word_encode_dim = word_encode_dim
        self.sent_encode_dim = sent_encode_dim

        in_tensor, out_tensor = self.build_network()

        super(han, self).__init__(inputs=in_tensor, outputs=out_tensor, name=name)

    def build_word_encoder(self, max_words, embed_matrix, encode_dim=200):
        vocab_size = embed_matrix.shape[0]
        embed_dim = embed_matrix.shape[1]
        embed_layer = keras.layers.Embedding(vocab_size, embed_dim, weights=[embed_matrix],
                                             input_length=max_words, trainable=False)
        sent_input = keras.layers.Input(shape=(max_words,), dtype="int32")
        embed_sents = embed_layer(sent_input)
        encode_sents = keras.layers.Bidirectional(keras.layers.GRU(int(encode_dim / 2)))(
            embed_sents)
        return keras.Model(inputs=[sent_input], outputs=[encode_sents], name="word_encoder")

    def build_sent_encoder(self, max_sents, summary_dim, encode_dim=200):
        text_input = keras.layers.Input(shape=(max_sents, summary_dim))
        encode_sents = keras.layers.Bidirectional(keras.layers.GRU(int(encode_dim / 2)))(
            text_input)
        return keras.Model(inputs=[text_input], outputs=[encode_sents], name="sentence_encoder")

    def build_network(self):
        in_tensor = keras.layers.Input(shape=(self.max_sents, self.max_words))
        word_encoder = self.build_word_encoder(self.max_words, self.embed_matrix, self.word_encode_dim)
        word_rep = keras.layers.TimeDistributed(word_encoder, name="word_encoder")(in_tensor)
        sentence_rep = keras.layers.TimeDistributed(han_attention_layer(), name="word_attention")(word_rep)
        doc_rep = self.build_sent_encoder(self.max_sents, self.word_encode_dim, self.sent_encode_dim)(sentence_rep)
        doc_summary = han_attention_layer(name="sentence_attention")(doc_rep)
        out_tensor = keras.layers.Dense(self.output_size, activation="softmax", name="class_prediction")(doc_summary)
        return in_tensor, out_tensor
    


# Load Kaggle IMDB Dataset
dataset = pd.read_csv("/users/duruoheng/Desktop/text_classification/nlp_han/Kaggle_data/labeledTrainData.tsv", sep="\t")
reviews = dataset["review"].values
sentiments = dataset["sentiment"].values
print(dataset.head())

# Tokenize
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(reviews)

# Input matrix for Model, zero-pad as to not affect predictions of attention mechanism
x = np.zeros((len(reviews), MAX_SENT, MAX_WORDS), dtype="int32")

for i, review in enumerate(reviews):

    # Separate each review into individual sentences
    # https://www.nltk.org/api/nltk.tokenize.html
    sentences = nltk.tokenize.sent_tokenize(review)
    tokenized_sents = tokenizer.texts_to_sequences(sentences)

    # Add padding
    tokenized_sents = keras.preprocessing.sequence.pad_sequences(tokenized_sents, maxlen = MAX_WORDS)
    padding = MAX_SENT - tokenized_sents.shape[0]

    # No padding needed
    if padding < 0:
        tokenized_sents = tokenized_sents[0:MAX_SENT]
    else:
        # Add padding
        tokenized_sents = np.pad(tokenized_sents, ((0, padding), (0, 0)), mode = 'constant', constant_values = 0)

    # Add to input matrix
    x[i] = tokenized_sents[None, ...]

# Convert sentiments for Keras
y = keras.utils.to_categorical(sentiments)

# Validation/Train 20/80 split
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
x_train = x[:4000]
y_train = y[:4000]
x_val = x[4000:4500]
y_val = y[4000:4500]
x_test = x[4500:5000]
y_test = y[4500:5000]

# Load GloVe Word Vectors Pretrained on Wikipedia data - 6B tokens, 400K vocab, uncased, 100d vectors
# https://nlp.stanford.edu/projects/glove/
file = open("/users/duruoheng/Desktop/text_classification/nlp_han/glove.6B/glove.6B.100d.txt", "r", encoding = "utf-8")
lines = file.readlines()
embeddings = dict()
for line in lines:
    vals = line.split()
    embeddings[vals[0]] = np.asarray(vals[1:], dtype="float32")

# Create weight matrix from embeddings
embed_matrix = np.random.random((len(tokenizer.word_index) + 1, GLOVE_DIM))
embed_matrix[0] = 0
for word, i in tokenizer.word_index.items():
    embed_vec = embeddings.get(word)
    if embed_vec is not None:
        embed_matrix[i] = embed_vec



# Train HAN Model
han_model = han(MAX_WORDS, MAX_SENT, 2, embed_matrix, word_encode_dim=100, sent_encode_dim=100)
han_model.summary()
han_model.compile(optimizer="adagrad", loss="categorical_crossentropy", metrics=["acc"])
history = han_model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val))
han_model.save('han_model.h5')

# Load the pre-trained model
# from keras.models import load_model
# model = han(load_model('han_model.h5'))
loss, acc = han_model.evaluate(x_val, y_val, verbose=0)
print('Test Accuracy: %f' % (acc*100))

# Generate accuracy/loss curves
fig1 = plt.figure()
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves: HAN',fontsize=16)
fig1.savefig('loss_han.png')
plt.show()

fig2=plt.figure()
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves: HAN',fontsize=16)
fig2.savefig('accuracy_han.png')
plt.show()

# library
from pathlib import Path
import re
import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
from numpy.linalg import norm
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


# directory & file
with open("bible.txt", "r") as file:
    data = file.read()

vocab_size = 10000
data = [data]  # data as list to be fed into tokenizer


# preprocess and tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ',
    char_level=False,
    oov_token=None)

tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)


# Obtain the word index (mapping between integers and words)
word_index = tokenizer.word_index


# Define a function to translate tokens to words
def translate_function(tokens, word_index):
    words = [list(word_index.keys())[list(word_index.values()).index(token)] for token in tokens]
    return words

# Translate the sequences back to words
# words = [translate_function(sequence, word_index) for sequence in sequences]


# create input target pairs
window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
      sequences[0],
      vocabulary_size=vocab_size,
      window_size=window_size,
      negative_samples=0)




class SkipGram(tf.keras.Model):
    def __init__(self,vocab_size,embedding_size):
        super().__init__()

        self.optimizer=tf.keras.optimizers.Adam(1e-4)
        self.loss_function=tf.keras.losses.CosineSimilarity()

        self.embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim = embedding_size,
            embeddings_initializer='uniform',
            mask_zero=False
        )

    def call(self,input):

        x = self.embedding(input)

        return x

    def train_step(self,data):

        with tf.GradientTape() as tape:

            input = data[0]
            target = data[1]

            input_embedding = self.call(input)

            target_embedding = self.call(target)

            loss = self.loss_function(target_embedding,input_embedding)

            # we calculate the negative loss since the optimizer will decrease the loss,
            # but since our cosine similarity should be increase as the model trains

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss




# Hyperparameters

epochs = 10
vocab_size = 10000
embedding_size = 64

hw_directory = str(Path(__file__).parents[0])

model = SkipGram(vocab_size,embedding_size)

loss_lst = []

positive_skip_grams = positive_skip_grams[:10000]

def training_loop(model,data):

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        # pair is one two-word list from the sequences list of lists (the input/target pairs)

        input_indices = []
        input_embeddings = []

        for pair in tqdm(data):
            loss = model.train_step(pair)
            loss_lst.append(loss)

        input_embeddings = model.embedding.weights[0]


    return loss_lst, input_embeddings


def test_embeddings(test_index, embedding_list, word_index):

    

    # embedding of text index
    test_embed = embedding_list[test_index]

    # cosine similarity between test_embed and all the embeddings
    cosine_sim = []

    for embed in embedding_list:
        if np.array_equal(embed,test_embed) != True:
            cosine = np.dot(test_embed, embed) / (norm(test_embed) * norm(embed))
            cosine_sim.append(cosine)

    # indices of top 10 values
    top_index = sorted(range(len(cosine_sim)), key=lambda i: cosine_sim[i], reverse=True)[:10]

    # translate index to words
    token_translator = translate_function(word_index.values(), word_index)

    test_word = token_translator[test_index]

    neighbors = []

    for i in top_index:
        neighbor = token_translator[i]
        neighbors.append(neighbor)

    return test_word, neighbors

def loss_plot(loss_lst):
    fig = plt.figure()
    line1, = plt.plot(loss_lst)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend([line1], ["Loss"])
    plt.title(f"Loss with {epochs} epochs")
    fig.savefig(f"plots/loss_w_{epochs}epochs.png")
    plt.show()

loss_lst, embeddings = training_loop(model, positive_skip_grams[:1000])

loss_plot(loss_lst)

test_word, sanity_check_list = test_embeddings(100, embeddings, word_index)

print(f'The 10 words closest to the target word - {test_word} - are {sanity_check_list}')



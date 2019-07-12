import tensorflow as tf
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
import math
from scipy import spatial
import pandas as pd
import csv


def one_hot_encoding(X_index, Y_index, totalvocab):

    X_hot = (np.arange(total_vocab+1) == X_index[..., None])
    Y_hot = (np.arange(total_vocab+1) == Y_index[..., None])

    np.save('X_hot.npy', X_hot)
    np.save('Y_hot.npy', Y_hot)


def load_training_data(x_name, y_name):
    X_train = np.load(x_name)
    Y_train = np.load(y_name)

    return X_train, Y_train


def make_dict_indice(word_to_indice):

    indice_to_word = {}

    for k, v in word_to_indice.items():
        indice_to_word[v] = k

    print(indice_to_word)
    # Save the dictionary
    with open('indice_to_word' + '.pkl', 'wb') as f:
        pickle.dump(indice_to_word, f, pickle.HIGHEST_PROTOCOL)


def make_dictionary():
    X_train, Y_train = load_training_data('./X_train.npy', './Y_train.npy')
    word_list = []
    word_to_indice = {}
    index = 0
    X_train_index = []
    Y_train_index = []

    # Accumulate all words into a word list
    for example in X_train:
        for word in example:
            word_list.append(word)

    for example in Y_train:
        for word in example:
            word_list.append(word)

    # Get only the unique words
    word_list = set(word_list)
    extra = len(word_list)

    # Give indices to words
    for w in word_list:
        if w not in word_to_indice:
            word_to_indice[w] = index
            index += 1

    # Convert words to indices in training data
    for example in X_train:
        temp = []
        for word in example:
            if word == '0':
                temp.append(extra)
            else:
                temp.append(word_to_indice[word])

        X_train_index.append(temp)

    for example in Y_train:
        temp = []
        for word in example:
           temp.append(word_to_indice[word])

        Y_train_index.append(temp)

    X_train_index = np.array(X_train_index)
    Y_train_index = np.array(Y_train_index).reshape(len(X_train_index), 1)

    print(X_train_index)
    print(Y_train_index)

    # Save the dictionary
    with open('word_to_indice' + '.pkl', 'wb') as f:
        pickle.dump(word_to_indice, f, pickle.HIGHEST_PROTOCOL)

    # Save the indexed training data
    np.save('X_index.npy', X_train_index)
    np.save('Y_index.npy', Y_train_index)


def load_dictionary(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Hyper parameters
word_to_indice = load_dictionary('word_to_indice')
indice_to_word = load_dictionary('indice_to_word')

X_index = np.load('X_index.npy')
Y_index = np.load('Y_index.npy')
m = len(X_index)
batch_size = 256
learning_rate = 0.0001
epochs = 100
e_features = 256
n1 = 1024
n2 = 512
n3 = 256
total_vocab = len(word_to_indice)

x = tf.placeholder('int64', [None, 4], name='input')  # Number of examples x features
y = tf.placeholder('int64', [None, 1], name='output')  # Number of examples x output


def forwardProp(x, y):

    # This is a word embedding matrix which will consists of features for every word, so feature length x total vocabulary size
    embedding_mat = tf.get_variable("EM", shape=[total_vocab, e_features], initializer=tf.random_normal_initializer(seed=1))

    # m x words x total_vocab * total_vocab x e_features = m x words x e_features
    # embed_x = tf.tensordot(x, tf.transpose(embedding_mat), axes=[[2], [0]])
    # embed_y = tf.tensordot(y, tf.transpose(embedding_mat), axes=[[2], [0]])

    embed_x = tf.gather(embedding_mat, x)  # m x words x e_features
    embed_y = tf.gather(embedding_mat, y)  # m x words x e_features

    #print("Shape of embed x", embed_x.get_shape())

    W1 = tf.get_variable("W1", shape=[n1, e_features], initializer=tf.random_normal_initializer(seed=1))
    B1 = tf.get_variable("b1", shape=[1, 4, n1], initializer=tf.zeros_initializer())

    # m x words x e_features *  e_features x n1 = m x words x n1
    Z1 = tf.add(tf.tensordot(embed_x, tf.transpose(W1), axes=[[2], [0]]), B1, )
    A1 = tf.nn.tanh(Z1)

    W2 = tf.get_variable("W2", shape=[n2, n1], initializer=tf.random_normal_initializer(seed=1))
    B2 = tf.get_variable("B2", shape=[1, 4, n2], initializer=tf.zeros_initializer())

    # m x words x n1 *  n1 x n2 = m x words x n2
    Z2 = tf.add(tf.tensordot(A1, tf.transpose(W2), axes=[[2], [0]]), B2)
    A2 = tf.nn.tanh(Z2)

    W3 = tf.get_variable("W3", shape=[n3, n2], initializer=tf.random_normal_initializer(seed=1))
    B3 = tf.get_variable("B3", shape=[1, 4, n3], initializer=tf.zeros_initializer())

    # m x words x n2  * n2 x n3 = m x words x n3
    Z3 = tf.add(tf.tensordot(A2, tf.transpose(W3), axes=[[2], [0]]), B3)
    A3 = tf.nn.tanh(Z3)

    # Convert m x words x n3 to m x n3
    x_final = tf.reduce_mean(A3, axis=1)
    y_final = tf.reduce_mean(embed_y, axis=1)

    return x_final, y_final


def backProp(X_index, Y_index):
    x_final, y_final = forwardProp(x, y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_final, labels=y_final))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    total_batches = math.floor(m/batch_size)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            batch_start = 0

            for i in range(int(m/batch_size)):

                x_hot = X_index[batch_start: batch_start + batch_size]
                y_hot = Y_index[batch_start: batch_start + batch_size]
                batch_start += batch_size

                _, temp_cost = sess.run([optimizer, cost], feed_dict={x: x_hot, y: y_hot})

                print("Cost at minibatch:  ", i , " and epoch ", epoch, " is ", temp_cost)

            if m % batch_size != 0:
                x_hot = X_index[batch_start: batch_start+m - (batch_size*total_batches)]
                y_hot = Y_index[batch_start: batch_start+m - (batch_size*total_batches)]
                _, temp_cost = sess.run([optimizer, cost], feed_dict={x: x_hot, y: y_hot})
                print("Cost at minibatch: (beyond floor)  and epoch ", epoch, " is ", temp_cost)


        # Saving the model
        save_path = saver.save(sess, "./model_neural_embeddingV2.ckpt")
        print("Model saved!")


def predict_search():

    # Initialize variables
    total_features = 4
    extra = len(word_to_indice)
    query = input('Enter your query')
    words = word_tokenize(query)
    # For now, it will throw an error if a word not present in dictionary is present
    features = [word_to_indice[w.lower()] for w in words]
    len_features = len(features)
    X_query = []
    Y_query = [[0]]  # Dummy variable, we don't care about the Y query while doing prediction
    if len_features < total_features:
        features += [extra] * (total_features - len_features)
    elif len_features > total_features:
        features = features[:total_features]

    X_query.append(features)
    X_query = np.array(X_query)
    print(X_query)
    Y_query = np.array(Y_query)

    # Load the model
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    X_final, Y_final = forwardProp(x, y)
    saver = tf.train.Saver()
    #X_final, Y_final = forwardProp(x, y)

    with tf.Session() as sess:
        sess.run(init_global)
        sess.run(init_local)
        saver.restore(sess, './model_neural_embeddingV2.ckpt')
        print("Model loaded")
        print("Loaded variables are: ")
        print(tf.trainable_variables())
        print(sess.graph.get_operations())
        embedMat = sess.run('EM:0')  # Get the word embedding matrix
        W1 = sess.run('W1:0')
        b1 = sess.run('b1:0')
        W2 = sess.run('W2:0')
        b2 = sess.run('B2:0')
        W3 = sess.run('W3:0')
        b3 = sess.run('B3:0')

        print(embedMat.shape)
        X_final = sess.run(X_final, feed_dict={x: X_query, y: Y_query})
        x_final = X_final.T

        print(x_final.shape)
        print(embedMat[0].shape)

        # Calculating using cosine similarity
        cos_mat = [spatial.distance.cosine(x_final, x) for x in embedMat]
        #
        # diff_mat = embedMat - x_final
        #
        # diff_mat = np.mean(diff_mat, axis=1)
        #
        sorted_mat = sorted(range(len(cos_mat)), key=lambda k: cos_mat[k])
        #
        print("\n\nRelated context words are (from forward prop):\n\n")

        for i in range(100):
            print(i, ":", indice_to_word[sorted_mat[i]], end=' ')
            if i%20 == 0:
                print("")

        print("\n\nDirect embedding matrix comparisons")
        index = word_to_indice[words[0]]
        cos_mat = [spatial.distance.cosine(embedMat[index], x) for x in embedMat]
        sorted_mat = sorted(range(len(cos_mat)), key=lambda k: cos_mat[k])

        for i in range(100):
            print(i, ":", indice_to_word[sorted_mat[i]], end=' ')
            if i%20 == 0:
                print("")
        '''
        
        # Shabby way of running the forward prop during test time 


        embedMat = np.vstack((embedMat, np.zeros((1, e_features))))

        embed_x = embedMat[X_query, :]

        Z1 = np.add(np.dot(embed_x, W1.T), b1)
        A1 = np.tanh(Z1)

        Z2 = np.add(np.dot(A1, W2.T), b2)
        A2 = np.tanh(Z2)

        Z3 = np.add(np.dot(A2, W3.T), b3)
        A3 = np.tanh(Z3)

        x_final = np.mean(A3, axis=1)
        print(x_final.shape)

        diff_mat = np.subtract(embedMat, x_final)

        diff_mat = np.mean(diff_mat, axis=1)


        sorted_mat = sorted(range(len(diff_mat)), key=lambda k: diff_mat[k])

        print("\n\nRelated context words are:\n\n")

        for i in range(50):
            print(indice_to_word[sorted_mat[i]])

        '''

def glove_mod(file):
        n = 20  # Number of closest words to show
        print("Loading Glove Model")
        with open(file, encoding="utf8") as f:
            content = f.readlines()
        model = {}
        for line in content:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.", len(model), " words loaded!")

        words = pd.read_table(file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

        print("Loaded")

        print("Enter query or press q to quit")
        reply = ""
        while reply != 'q':
            reply = input("query: ").lower()
            to_find = model[reply]

            for w in range(n):
                diff = words.as_matrix() - to_find  # Difference with respect to all words
                delta = np.sum(diff * diff, axis=1)  # The difference squared reduced to a single value for each word
                index = np.argmin(delta)
                print(words.iloc[index].name)
                words = words.drop(words.iloc[index].name, axis=0)









#backProp(X_index, Y_index)

#one_hot_encoding(X_index, Y_index, total_vocab)
# print(X_index.shape)
# print(Y_index.shape)
#backProp(X_index, Y_index)
predict_search()
#glove_mod('./glove.6B.50d.txt')

#make_dict_indice(word_to_indice)

# X_train = np.load('X_train.npy')
# Y_train = np.load('Y_train.npy')
#
# for i in range(len(X_train)):
#     print(X_train[i], Y_train[i])



import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import process_data


def make_training_data(titles):

    total_features = 4
    X = []
    Y = []

    # Go through the articles (as the titles list consists of three pandas series), which consists of the titles
    for articles in titles:
        for title in articles:
            # To make sure it is a string
            if isinstance(title, str):
                # Tokenize the title
                tokenized = word_tokenize(title)

                # To make the data simpler, grab only the nouns, also make them lowercase
                noun_words = [word.lower() for (word, pos) in nltk.pos_tag(tokenized) if pos.startswith('NN')]

                # Get ride of ’s and ’ at the end
                words = []
                for w in noun_words:
                    if '’s' in w:
                        temp = w.strip('’s')
                        if len(temp) >= 1:
                            words.append(temp)
                    elif '’' in w:
                        temp = w.strip('’')
                        if len(temp) >= 1:
                            words.append(temp)
                    else:
                        words.append(w)

            # Make sure, words array consist of at least one example

            if len(words) >= 1:

                # Limit the input features to 4 and take the last word as a dummy output
                Y.append(words[-1])

                # Pad the input features if they are lesser than or greater than 4
                features = words[:-1]
                len_features = len(features)

                if len_features < total_features:
                    features += [0] * (total_features - len_features)
                elif len_features > total_features:
                    features = features[:total_features]

                X.append(features)

    return X, Y


def save_data(X_train, Y_train):

    # Shuffle training data
    m = len(X_train)  # Total number of training examples
    permut = np.random.permutation(m)
    X_train = X_train[permut, :]
    Y_train = Y_train[permut, :]

    np.save('X_train.npy', X_train)
    np.save('Y_train.npy', Y_train)


if __name__ == '__main__':
    # Only take the titles of the data, otherwise it will become too humongous to train on a NN
    titles = process_data.load_headers(['articles1.csv', 'articles2.csv', 'articles3.csv'])
    X, Y = make_training_data(titles)

    X_train = np.array(X)
    Y_train = np.array(Y)

    print(X_train.shape)
    print(Y_train.shape)

    print(X_train)
    print(Y_train)

    save_data(X_train, Y_train.reshape(len(X_train), 1))





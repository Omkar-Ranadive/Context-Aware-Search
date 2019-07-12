import pandas as pd
import process_data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
import itertools


def create_stop_list():
    f = open('stopwords.txt', 'r')
    stopList = [w.strip() for w in f.readlines()]

    return stopList


def create_prob_dist(titles, contents):
    word_dist = {}
    word_dist_final = {}
    stopList = create_stop_list()
    counter = 0

    # The titles list consist of three panda series, each obtained from different csv file
    # Each panda series consist of all titles of that particular csv
    for articles in titles:
        for title in articles:

            # This is to get rid of incomplete data which is not in form of strings
            if isinstance(title, str):
                # Get rid of stopwords
                words = word_tokenize(title)
                words = [w.lower() for w in words if w.lower() not in stopList]

                # Get all possible combinations of words
                adjacentWords = itertools.product(words[:-1], words[1:])

                for cur, next in adjacentWords:

                    # The cartesian product results in tuples having same words. So, ignore those.
                    if cur != next:
                        # Format current_word = {next_word: number_of_occurrences}

                        # If the current word(first) doesn't exist in word dist, then create an entry
                        if cur not in word_dist:
                            word_dist[cur] = {next:1}

                        # If the current word exists but not the adjacent, then create an entry for it
                        elif next not in word_dist[cur]:
                            word_dist[cur][next] = 1

                        # If word and next word exists, increase the count
                        else:
                            word_dist[cur][next]+=1

                counter+=1
                if counter % 10000 == 0:
                    print("Total titles processed: ", counter)

        # Repeat the process for contents list
        # Commented out for the time being, as it requires large amount of RAM ( > 16 GB at least)
    '''
     for articles in contents:
        for content in articles:
            # This is to get rid of incomplete data which is not in form of strings
            if isinstance(content, str):
                # Get rid of stopwords
                words = word_tokenize(content)
                words = [w.lower() for w in words if w.lower() not in stopList]

                # Get all possible combination of words
                adjacentWords = itertools.product(words[:-1], words[1:])

                for cur, next in adjacentWords:

                    if cur != next:
                        # Format current_word = {next_word: number_of_occurrences}

                        # If the current word(first) doesn't exist in word dist, then create an entry
                        if cur not in word_dist:
                            word_dist[cur] = {next: 1}

                        # If the current word exists but not the adjacent, then create an entry for it
                        elif next not in word_dist[cur]:
                            word_dist[cur][next] = 1

                        # If word and next word exists, increase the count
                        else:
                            word_dist[cur][next] += 1
    '''

    # Now calculate the probability distribution from the total number of counts
    for cur, next_words in word_dist.items():
        word_dist_final[cur] = {}

        # Total count of number of occurrences of then next possible words after cur
        total_count = sum(next_words.values())

        for w in next_words:
            prob = next_words[w]/total_count
            word_dist_final[cur][w] = prob

    print(len(word_dist_final))

    return word_dist_final


def create_prob_dist_2gram(titles, contents):
    word_dist = {}
    word_dist_final = {}
    stopList = create_stop_list()
    counter = 0

    # The titles list consist of three panda series, each obtained from different csv file
    # Each panda series consist of all titles of that particular csv
    for articles in titles:
        for title in articles:

            # This is to get rid of incomplete data which is not in form of strings
            if isinstance(title, str):
                # Get rid of stopwords
                words = word_tokenize(title)
                words = [w.lower() for w in words if w.lower() not in stopList]

                # Create a 2-gram, i.e a sequence of three words
                adjacentWords = zip(words[0:], words[1:], words[2:])

                for first, second, third in adjacentWords:

                    # The key will be a tuple of first two words
                    key = (first, second)
                    # The cartesian product results in tuples having same words. So, ignore those.

                    if key not in word_dist:
                        word_dist[key] = {third: 1}

                    # If the current word exists but not the adjacent, then create an entry for it
                    elif third not in word_dist[key]:
                        word_dist[key][third] = 1

                    # If word and next word exists, increase the count
                    else:
                        word_dist[key][third] += 1

                counter += 1
                if counter % 10000 == 0:
                    print("Total titles processed: ", counter)

    # Now calculate the probability distribution from the total number of counts
    # Note, over here cur will be a tuple of two words (w1, w2)
    for cur, next_words in word_dist.items():
        word_dist_final[cur] = {}

        # Total count of number of occurrences of then next possible words after cur
        total_count = sum(next_words.values())

        for w in next_words:
            prob = next_words[w] / total_count
            word_dist_final[cur][w] = prob

    print(len(word_dist_final))

    return word_dist_final


if __name__ == '__main__':

    # Form prob dist
    titles, contents = process_data.load_files(['articles1.csv', 'articles2.csv', 'articles3.csv'])
    prob_dist = create_prob_dist_2gram(titles, contents)

    # Save the prob distribution
    with open('prob_dist_2gram.pickle', 'wb') as handle:
        pickle.dump(prob_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)



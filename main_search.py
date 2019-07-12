import pickle
import nltk
from nltk.tokenize import word_tokenize


# Load the saved prob distribution file
with open('prob_dist.pickle', 'rb') as handle:
    prob_dist = pickle.load(handle)


# Load the two gram model
with open('prob_dist_2gram.pickle', 'rb') as handle:
    prob_dist_2gram = pickle.load(handle)

print("Length of 1 gram distribution", len(prob_dist))
print("Length of 2 gram distribution: ", len(prob_dist_2gram))

#print(prob_dist_2gram.keys())


def search():
    query = input("Enter search query: ")
    words = word_tokenize(query)
    #key = (words[0].lower(), words[1].lower())


    print("Related context terms are: ")

    '''
    if key in prob_dist_2gram:
        print("Two gram search: ")
        context_w = sorted(prob_dist_2gram[key], key=prob_dist_2gram[key].get, reverse=True)
        counter = 0
        for c in context_w:
            print(c, end=' , ')
            counter += 1
            if counter % 20 == 0:
                print("")
    '''
    print("\n\n1 gram search: ")
    for w in words:
        print("Word: ", w)

        # Check if word exists in dictionary
        if w.lower() in prob_dist:
            print("Related words: (Similar context)")

            # Sort dictionary first, to get related context in proper order
            context_w = sorted(prob_dist[w.lower()], key=prob_dist[w.lower()].get, reverse=True)
            print("Context length", len(context_w))
            related_nouns = [word for (word, pos) in nltk.pos_tag(context_w) if pos[:2] == 'NN']

            # for n in context_w:
            #     print(nltk.pos_tag(context_w))


            print("\n\nRelated noun", len(related_nouns))

            counter = 0
            #print(related_nouns)
            for c in related_nouns:
                print(c, end=' , ')
                counter += 1
                if counter%20 == 0:
                    print("")

            print("\n\n------------------------------------------------------------------\n\n")


            counter = 0
            for c in context_w:
                print(c, end=' , ')
                counter += 1
                if counter % 20 == 0:
                    print("")

        else:
            print("Word doesn't exist in dictionary. ")


search()


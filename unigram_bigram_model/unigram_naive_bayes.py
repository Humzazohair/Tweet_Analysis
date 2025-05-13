import reader
import math
from collections import Counter
from typing import List

def load_data(training_dir, test_dir, validation_dir):
    train_labels, train_text, train_id = reader.read_tsv_file(training_dir, has_header= True)
    test_labels, test_text, test_id  = reader.read_tsv_file(test_dir, has_header = False)
    validation_labels, validation_text, validation_id = reader.read_tsv_file(validation_dir, has_header = False)
    print(f"Training labels: {Counter(train_labels)}")
    print(f"Test labels: {Counter(test_labels)}")
    return train_labels, train_text, test_labels, test_text, test_id, validation_labels, validation_text, validation_id

def naive_bayes(train_labels, train_text, test_text, laplace=1, pos_prior=0.5):

    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

    bags = bag_seperator(train_labels, train_text)
    positive_bag = bags[0]
    negative_bag = bags[1]


    # Creates a hashmap, key = word, value = count of word
    positive_counter = Counter(positive_bag)
    negative_counter = Counter(negative_bag)
    

    positive_map = {}   # key = word, value = P(W = w | Y = Positive)
    negative_map = {}   # key = word, value = P(W = w | Y = Negative)

    # Create a combined set of unique words
    pos_set = set(positive_counter.keys())
    neg_set = set(negative_counter.keys())
    combined_set = pos_set.union(neg_set)

    # Calc P(W = w | Y = Positive) with laplace smoothing, fill out hashmaps
    calc_laplace(test_text, positive_map, positive_counter, combined_set, laplace)
    calc_laplace(test_text, negative_map, negative_counter, combined_set, laplace)

    # Calculate probability of P(Y = Positive/Negative | X = words)
    probabilities = []
            
    for i, doc in enumerate(test_text):
        pos_prob = math.log10(pos_prior)          
        neg_prob = math.log10(1 - pos_prior)
        for word in test_text[i]:
            pos_prob += math.log10(positive_map.get(word,1e-10))
            neg_prob += math.log10(negative_map.get(word,1e-10))

        probabilities.append([pos_prob, neg_prob])
    return probabilities


# HELPER FUNCTION:
# Takes in training data and seperates the data into two "bags" of positive words and negative words
def bag_seperator(train_labels, train_text)->List[List[str]]:
    positive_bag = []
    negative_bag = []
    res = []

    for i in range(len(train_labels)):
        for word in train_text[i]:
            if train_labels[i] == "INFORMATIVE":
                positive_bag.append(word)
            else:
                negative_bag.append(word)
                
    res.append(positive_bag)
    res.append(negative_bag)

    return res


# HELPER FUNCTION:
# Fills in a hashmap with P(W = w | Y = Positive/Negative)
# These values can be later used to calculate whether a review is Positive or Negative
def calc_laplace(test_text, hashmap, counter, combined_set, k):
    summation = sum(counter.values()) + k * len(combined_set)
    for i in range(len(test_text)):
        for word in test_text[i]:
            word_cardinality = counter.get(word,0)
            hashmap[word] = (k + word_cardinality)/(summation)
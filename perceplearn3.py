import os
import re
import fnmatch
import sys
import numpy as np
from collections import Counter
import string

def search_files(directory='.', extension=''):

    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, extension):
            if 'README' not in filename:
                matches.append(os.path.join(root, filename))
    return matches


txtfiles = search_files(directory=sys.argv[1], extension="*.txt")

stop_word_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", ""]

filelist = []
vocabulary = []

for filename in txtfiles:
    filelist.append(filename)
    file = open(filename, 'r')
    text = file.read()
    file.close()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = re.split(r'\W+', text)
    for word in words:
        if word.lower() not in stop_word_list:
            vocabulary.append(word.lower())


cnt = Counter(vocabulary)

vocab = []
for i in cnt:
    if 700 > cnt[i] > 11:
        vocab.append(i)
vocab=list(sorted(vocab))
## Positive-Negative class ##


permutation_list = []

for filename in filelist:
    if 'positive' in filename:
        Y = 1
    elif 'negative' in filename:
        Y = -1
    Xi = []
    file = open(filename, 'r')
    text = file.read()
    file.close()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = re.split(r'\W+', text)
    words = [x.lower() for x in words]

    for word in vocab:
        Xi.append(words.count(word))
    pair = (Xi,Y)

    permutation_list.append(pair)

# Vanilla perceptron #
W = np.zeros(len(vocab))
b = 0
total = 0
for epoch in range(20):
    permutation_list = np.random.permutation(permutation_list)
    for i in range(len(permutation_list)):
        total = np.dot(permutation_list[i][0], W)
        a = total + b
        Y_actual = permutation_list[i][1]
        if Y_actual * a <= 0:
            W += np.dot(Y_actual, permutation_list[i][0])
            b += Y_actual

W_van_posneg = W
b_van_posneg = b

# Averaged perceptron #
W = np.zeros(len(vocab))
U = np.zeros(len(vocab))
b = 0
β = 0
c = 0
for epoch in range(20):
    permutation_list = np.random.permutation(permutation_list)
    for i in range(len(permutation_list)):
        Y_actual = permutation_list[i][1]
        if Y_actual * (np.dot(W, permutation_list[i][0]) + b) <= 0:
            W += np.dot(Y_actual , permutation_list[i][0])
            b += Y_actual
            U += np.dot(Y_actual * c , permutation_list[i][0])
            β += Y_actual * c
        c += 1

W_avg_posneg = W - U/c
b_avg_posneg = b - β/c

## Truthful-Deceptive class ##

W = np.zeros(len(vocab))
b = 0
permutation_list = []

for filename in filelist:
    if 'truthful' in filename:
        Y = 1
    elif 'deceptive' in filename:
        Y = -1
    Xi = []
    file = open(filename, 'r')
    text = file.read()
    file.close()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = re.split(r'\W+', text)
    words = [x.lower() for x in words]

    for word in vocab:
        Xi.append(words.count(word))
    pair = (Xi,Y)

    permutation_list.append(pair)

# Vanilla perceptron #
total = 0
for epoch in range(20):
    permutation_list = np.random.permutation(permutation_list)
    for i in range(len(permutation_list)):
        total = np.dot(permutation_list[i][0], W)
        a = total + b
        Y_actual = permutation_list[i][1]
        if Y_actual * a <= 0:
            W += np.dot(Y_actual, permutation_list[i][0])
            b += Y_actual

W_van_truthdec = W
b_van_truthdec = b

# Averaged perceptron #
W = np.zeros(len(vocab))
U = np.zeros(len(vocab))
b = 0
β = 0
c = 1
for epoch in range(20):
    permutation_list = np.random.permutation(permutation_list)
    for i in range(len(permutation_list)):
        Y_actual = permutation_list[i][1]
        if Y_actual * (np.dot(W, permutation_list[i][0]) + b) <= 0:
            W += np.dot(Y_actual , permutation_list[i][0])
            b += Y_actual
            U += np.dot(Y_actual * c , permutation_list[i][0])
            β += Y_actual * c
        c += 1

W_avg_truthdec = W - U/c
b_avg_truthdec = b - β/c


output = open ('vanillamodel.txt' , 'w')

output.write(str(b_van_posneg) + '\n' + str(b_van_truthdec) + '\n' + str(len(W_van_posneg)) + '\n')
for i in range(len(W_van_posneg)):
    output.write(str(W_van_posneg[i]) + '\n')
for j in range(len(W_van_truthdec)):
    output.write(str(W_van_truthdec[j]) + '\n')
for word in vocab:
    output.write(word +  '\n')

output.close


output = open ('averagedmodel.txt' , 'w')

output.write(str(b_avg_posneg) + '\n' + str(b_avg_truthdec) + '\n' + str(len(W_avg_posneg)) + '\n')
for i in range(len(W_avg_posneg)):
    output.write(str(W_avg_posneg[i]) + '\n')
for j in range(len(W_avg_truthdec)):
    output.write(str(W_avg_truthdec[j]) + '\n')
for word in vocab:
    output.write(word +  '\n')

output.close

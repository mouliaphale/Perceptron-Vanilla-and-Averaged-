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


txtfiles = search_files(directory=sys.argv[2], extension="*.txt")
model = sys.argv[1]


# Reading data from the file
W_posneg = []
W_truthdec = []
vocab = []
openfile = open(model,'r')
b_posneg = float(openfile.readline())
b_truthdec = float(openfile.readline())
length_of_vector = int(openfile.readline())
for i in range(length_of_vector):
    W_posneg.append(float(openfile.readline()))
for i in range(length_of_vector):
    W_truthdec.append(float(openfile.readline()))
for line in openfile:
    vocab.append(line.strip())
vocab=list(sorted(vocab))


x = []
Y = []
for filename in txtfiles:
    Xd = []
    file = open(filename, 'r')
    text = file.read()
    file.close()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = re.split(r'\W+', text)
    words = [x.lower() for x in words]

    for word in vocab:
        Xd.append(words.count(word))

    x.append(Xd)

total = 0
answer = {}

for i in range(len(x)):
    total = np.dot(x[i], W_posneg)
    a = total + b_posneg
    prediction_posneg = np.sign(a)

    total = np.dot(x[i], W_truthdec)
    a = total + b_truthdec
    prediction_truthdec = np.sign(a)

    predictions = []
    if prediction_truthdec == 1:
        predictions.append('truthful')
    else:
        predictions.append('deceptive')
    if prediction_posneg == 1:
        predictions.append('positive')
    else:
        predictions.append('negative')
    answer[txtfiles[i]] = predictions

output = open ('percepoutput.txt' , 'w')

for a in answer:
    output.write(str(answer[a][0]) + '\t' + str(answer[a][1]) + '\t' + str(a) + '\n')

output.close

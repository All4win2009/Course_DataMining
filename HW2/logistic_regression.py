# There are 2 classes and 11392-dimensional features for each sample.
# There are 2177020 training examples and 220245 testing examples.
import numpy as np
import csv
import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def stochGradAscent(data_matrix, class_labels, num_iter=100):
    m = 2177020
    #m = 20000
    n = 11392
    # m rows = m datas
    # n cols = n dims
    weights = np.zeros(n)
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            if i % 1000 == 0:
                print'num: ', i, ' is dealing'
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(data_index)))
            h = sigmoid( weights[ data_matrix[randIndex] ].sum() )
            error = class_labels[randIndex] - h
            weights[ data_matrix[randIndex] ] += alpha * error
            del (data_index[randIndex])
    return weights


dataMatrix = []
classLabels = []
data_file = open('/Users/All4win/Documents/Three/DataMining/HW2/train_data.txt', 'rb')
reader = csv.reader(data_file, delimiter=',')
index = 0
for row in reader:
	if index < 2177020:
		classLabels.append(int(row[0]))
		dataMatrix.append(row[1:])
		index = index + 1
	else:
		break

result = stochGradAscent(dataMatrix, classLabels, 1)
csvFile = open("/Users/All4win/Documents/Three/DataMining/HW2/weight.csv", 'wb')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(result)
csvFile.close()

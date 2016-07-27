import numpy as np
import csv
import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

data_file = open('/Users/All4win/Documents/Three/DataMining/HW2/cpp_weight.txt', 'r')
reader = csv.reader(data_file, delimiter=',')

weight = []
for row in reader:
	for item in row:
		weight.append(float(item))

test_data = []
test_file = open('/Users/All4win/Documents/Three/test.txt', 'r')
test_reader = csv.reader(test_file, delimiter = ' ')
for row in test_reader:
    test_data.append(row[1:])

index = 0
result = []
for row in test_data:
    value = 0
    for item in row:
        value += weight[ int(item.split(':')[0]) - 1 ]
    h = sigmoid(value)
    if h >=0.5:
    	h = 1
    else:
    	h = 0
    line = str(index) + ',' + str(h) +'\n'
    result.append(line)
    index += 1

file = open("/Users/All4win/Documents/Three/DataMining/HW2/result_3.txt", 'wb')
file.writelines(result)
file.close()
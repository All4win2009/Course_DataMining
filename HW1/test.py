import numpy as np
import csv

#feature vector init theta0 - theta 384
feature = np.genfromtxt('feature_list.csv', delimiter=",")

#temp feature
temp = feature

#load test_data
data = np.genfromtxt('test.csv', delimiter=",")
test_data = data[1:, 1:]

result = np.dot( test_data, feature[1:]) + feature[0]

output = []
index = 0
for line in result:
	temp = [index, line]
	output.append(temp)
	index = index + 1

csvFile = open("submission.csv", 'wb')
csvWriter = csv.writer(csvFile)
csvWriter.writerows(output)
csvFile.close()
import numpy as np
import csv

#learning rate
a = 0.06

#num of feature
n = 385

#num of train data
m = 25000

#feature vector init theta0 - theta 384
# feature = np.genfromtxt('feature_list.csv', delimiter=",")

feature = [0 for i in range(385)]

#temp feature
temp = feature

#load train_data
data = np.genfromtxt('train.csv', delimiter=",")
reference_list = data[1:, -1]
train_data = data[1:, 1:-1]

cost = 0
#  j:0-n
while True:
	difference = np.dot( train_data, feature[1:]) + feature[0] - reference_list
	precost = np.sum(difference ** 2)/ (2 * train_data.shape[0])
	j = 0
	while j < n:
		if j == 0:
			temp_sum = np.sum(difference) 
		else:
			temp_sum = np.sum(difference * train_data[:, j-1] )
		temp[j] = feature[j] - ( a / m ) * temp_sum
		j = j + 1
	difference = np.dot( train_data, temp[1:]) + temp[0] - reference_list
	cost = np.sum(difference ** 2)/ (2 * train_data.shape[0])
	
	feature = temp
	print 'Round: ',counter,'       Cost ',cost
	if counter % 50 == 0:
		csvFile = open("feature_list.csv", 'wb')
		csvWriter = csv.writer(csvFile)
		csvWriter.writerow(feature)
		csvFile.close()
	counter = counter + 1

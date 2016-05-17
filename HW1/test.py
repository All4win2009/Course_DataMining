import numpy as np
import csv

#learning rate
a = 0.1

#num of feature
n = 385

#num of train data
m = 25000

#feature vector init theta0 - theta 384
feature = np.genfromtxt('avEncode_list.csv', delimiter=",")

# feature = [0 for i in range(385)]

#temp feature
temp = feature

#load train_data
data = np.genfromtxt('train.csv', delimiter=",")
reference_list = data[1:, -1]
train_data = data[1:, 1:-1]


#counter:100 times  j:0-n  i:1-m
counter = 0
precost = 43
while True:
	
	
	difference = np.dot( train_data, feature[1:]) + feature[0] - reference_list
	cost = np.sum(difference ** 2)/ (2 * train_data.shape[0])
	if cost < precost:
		j = 0
		while j < n:
			if j == 0:
				temp_sum = np.sum(difference) 
			else:
				temp_sum = np.sum(difference * train_data[:, j-1] )
			temp[j] = feature[j] - ( a / m ) * temp_sum
			j = j + 1
	
			
		feature = temp
		print 'Round: ',counter,'       Cost ',cost
		csvFile = open("avEncode_list.csv", 'wb')
		csvWriter = csv.writer(csvFile)
		csvWriter.writerow(feature)
		csvFile.close() 
		precost = cost
		counter = counter + 1
	else:
		break

import numpy as np
import csv
from pybrain.structure import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
import random

m = 25000
# build network
fnn = buildNetwork(384, 100, 30, 1, bias=True)

# load train_data
data = np.genfromtxt('train.csv', delimiter=",")
reference_list = data[1:, -1]
train_data = data[1:, 1:-1]

DS = SupervisedDataSet(384,1)
for i in range(m):
	DS.addSample(train_data[i, :] , reference_list[i])

X = DS['input']
Y = DS['target']

# dataTrain, dataTest = DS.splitWithProportion(0.8)
# xTrain, yTrain = dataTrain['input'], dataTrain['target']
# xTest, yTest = dataTest['input'], dataTest['target']

trainer = BackpropTrainer(fnn, DS, momentum = 0.1, verbose = True, weightdecay = 0.01)
trainer.trainUntilConvergence(maxEpochs=100)

# load train_data
test_csv = np.genfromtxt('test.csv', delimiter=",")
test_data = test_csv[1:, 1:]

# c = random.randint(0, xTest.shape[0])
# X2 = xTest[c,:]

ans = []
for i in range(test_data.shape[0]):
	X2 = test_data[i,:]
	prediction = fnn.activate(X2)
	prediction.insert(0,i)
	ans.append(prediction)

csvFile = open("submission_2.csv", 'wb')
csvWriter = csv.writer(csvFile)
csvWriter.writerows(ans)
csvFile.close()

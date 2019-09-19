# Maya Z Shanmugam: I have adhered to the Honor Code in this Assignment 
import csv
import sys
import random
import math
import operator

#function that calculates Euclidan Distance.
#parameters: two instances and length
def euclideanDistance(v1, v2):
    distance = 0
    for x in range(1, len(v1)):
        distance += (v1[x] - v2[x])**2
    return math.sqrt(distance)

#function that calculates Hamming Distance 
def hammingDistance(s1, s2):
    diffs = 0
    for ch1, ch2 in zip(s1, s2):
       if ch1 != ch2:
           diffs += 1
    return diffs

#function that reads a file and divided the data between the training set and test set
def getDataSet (filename, seed, percentTraining, trainingSet =[], testSet = []):
    dataset = []
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        # getting column names from first row of file
        dataset = list(lines)
        print 'Original Length: ' + str(len(dataset))
        header = dataset[0]
        print 'Header: ' + str(header)
        dataset.pop(0)
        print 'Length After pop: ' + str(len(dataset))
        random.seed(seed)
        print 'Seed:  ' + str(seed)
        shuffled = list(dataset)
        print 'Length Shuffled: ' + str(len(shuffled))
        random.shuffle(shuffled)
        divide = int(float(len(dataset))*float(percentTraining))
        '''trainingSet = shuffled[:divide]
        testSet = shuffled[divide:] '''
        for i in range(len(shuffled)):
            for j in range(1, len(shuffled[i])):
                shuffled[i][j] = float(shuffled[i][j])   
            if len(trainingSet) < divide:
                trainingSet.append(shuffled[i])
            else:
                testSet.append(shuffled[i]) 
        # tests if header is pulled out of list                 
        print repr(len(dataset)) +' ' +repr(percentTraining)  + ' ' +repr(divide)
#function that reads a file and divided the data between the training set and test set
def getDataSetx (filename, seed, percentTraining, trainingSet =[], testSet = []):
    dataset = []
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        # getting column names from first row of file
        dataset = list(lines)
        print 'Original Length: ' + str(len(dataset))
        header = dataset[0]
        print 'Header: ' + str(header)
        dataset.pop(0)
        print 'Length After pop: ' + str(len(dataset))
        random.seed(seed)
        print 'Seed:  ' + str(seed)
        shuffled = list(dataset)
        print 'Length Shuffled: ' + str(len(shuffled))
        random.shuffle(shuffled)
        divide = int(float(len(dataset))*float(percentTraining))
        '''trainingSet = shuffled[:divide]
        testSet = shuffled[divide:] '''
        for i in range(len(shuffled)): 
            if len(trainingSet) < divide:
                trainingSet.append(shuffled[i])
            else:
                testSet.append(shuffled[i]) 
        # tests if header is pulled out of list                 
        print repr(len(dataset)) +' ' +repr(percentTraining)  + ' ' +repr(divide)
#function that returns a list of neighbors to the test instance provided        
def getNeighborsE(trainingSet, testInstance, k):
    distances = []
    for i in range(1,len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[i])
        #print 'Distance Euclidean' + repr(i) + ': ' + repr(dist)
        distances.append((trainingSet[i], dist))
    distances.sort(key=operator.itemgetter(1))
    #generates list of k nearest neighbors
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def getNeighborsH(trainingSet, testInstance, k):
    distances = []
    for i in range(1, len(trainingSet)):
        dist = hammingDistance(testInstance, trainingSet[i])
        #print 'Distance Hamming' + repr(i) + ': ' + repr(dist)
        distances.append((trainingSet[i], dist))
    distances.sort(key=operator.itemgetter(1))
    #generates list of k nearest neighbors
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors
'''
#returns list of votes 
def getResponse(neighbors):
	votes = {}
	for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in votes:
            votes[response] += 1
        else:
			votes[response] = 1
	sortedVotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
        print 'Sorted Votes: ' + repr(sortedVotes)
        print 'Votes: ' + repr(votes)
	return sortedVotes[0][0]'''
    
def getResponse(neighbors):
	votes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][0]
		if response in votes:
			votes[response] += 1
		else:
			votes[response] = 1
	sortedVotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
        #print 'Sorted Votes: ' + repr(sortedVotes)
        #print 'Votes: ' + repr(votes)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][0] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
    #load in parameters from terminal
    inputFile = sys.argv[1]
    distanceFunction = sys.argv[2]
    k = int(sys.argv[3])
    percentTrainingSet = sys.argv[4]
    randomSeed = int(sys.argv[5])
    
   #use parameters to process the data set 
    trainingSet=[]
    testSet=[]
    
    #do predictions
    predictions=[]
    if distanceFunction == 'H':
        getDataSetx(inputFile, randomSeed, percentTrainingSet, trainingSet, testSet)
        for i in range(len(testSet)):
           neighbors = getNeighborsH(trainingSet, testSet[i], k)
           result = getResponse(neighbors)
           predictions.append(result)
           print('> predicted=' + repr(result) + ', actual=' + repr(testSet[i][0]))
        accuracy = getAccuracy(testSet, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
    elif distanceFunction == 'E':
        getDataSet(inputFile, randomSeed, percentTrainingSet, trainingSet, testSet)
        for i in range(len(testSet)):
           neighbors = getNeighborsE(trainingSet, testSet[i], k)
           result = getResponse(neighbors)
           predictions.append(result)
           print('> predicted=' + repr(result) + ', actual=' + repr(testSet[i][0]))
        accuracy = getAccuracy(testSet, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
    
    
    '''print 'Train: ' + repr(len(trainingSet))
    print 'Test: ' + repr(len(testSet)) '''
    '''data1 = [2, 2, 2]
    data2 = [4, 4, 4]
    distance = euclideanDistance(data1, data2)
    hdistance = hammingDistance(data1, data2)
    print 'Euclidean Distance: ' + repr(distance)
    print 'Hamming Distance: ' + repr(hdistance)  
    trainSet = [[2, 2, 2], [4, 4, 4]]
    testInstance = [5, 5, 5]
    neighbors = getNeighborsE(trainSet, testInstance, 1)
    print(neighbors) '''
    '''neighbors = [['a',1,1,1], ['a',2,2,2], ['b',3,3,3]]
    response = getResponse(neighbors)
    print('response: ' + repr(response))
    testSet = [['a',1,1,1], ['a',2,2,2], ['b',3,3,3]]
    predictions = ['a', 'a', 'a']
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy :' + repr(accuracy))'''
    '''
    if distanceFunction == 'E':
        neighbors = getNeighborsE(trainingSet, testSet, k)
    elif distanceFunction == 'H':
        neighbors = getNeighborsH(trainingSet, testSet, k)
     '''   
    #testing sys args
    print('Inputs names are:' + inputFile + ' ' + distanceFunction + ' ' + repr(k) +' ' + percentTrainingSet + ' ' + repr(randomSeed) )
  
        
main()

                    
                    
    
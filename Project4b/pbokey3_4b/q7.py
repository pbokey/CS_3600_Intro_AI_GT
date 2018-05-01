import Testing
import NeuralNetUtil
import NeuralNet

num_neuron = 0
nnet_nohidden, testAccuracy_nohidden = NeuralNet.buildNeuralNet(Testing.xorData, maxItr = 200)
print "Accuracy (No Hidden Layer): ", testAccuracy_nohidden

while True:
	print "--------------num neuron:", num_neuron , " in the hidden layer------------------"
	i = 1
	acclist = []
	while i <= 5:
		print "Iteration Number:", i
		nnet, testAccuracy = NeuralNet.buildNeuralNet(Testing.xorData,maxItr = 200, hiddenLayerList = [num_neuron])
		acclist.append(testAccuracy)
		i += 1

	print "Final Calculations"
	print "Accuracy Average:", Testing.average(acclist)
	print "Standard Deviation:", Testing.stDeviation(acclist)
	print "Maximum:", max(acclist)
	if Testing.average(acclist) == 1:
		break
	num_neuron = num_neuron + 1
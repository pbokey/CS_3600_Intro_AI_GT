import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice

class Perceptron(object):
    """
    Class to represent a single Perceptron in the net.
    """
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights
    
    def getWeightedSum(self, inActs):
        """
        Returns the sum of the input weighted by the weights.
        
        Inputs:
            inActs (list<float/int>): input values, same as length as inSize
        Returns:
            float
            The weighted sum
        """
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])
    
    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the sigmoid function parametrized by 
            the value.
        """
        """YOUR CODE"""
        value = -(value)
        val_calc = 1 + exp(value)
        return 1.0 / val_calc
      
    def sigmoidActivation(self, inActs):                                       
        """
        Returns the activation value of this Perceptron with the given input.
        Same as g(z) in book.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            float
            The value of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        alias = inActs[:]
        newActs = [1.0] + alias
        weight_sum = self.getWeightedSum(newActs)
        return self.sigmoid(weight_sum)
        
    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the derivative of a sigmoid function
            parametrized by the value.
        """
        """YOUR CODE"""
        sig = self.sigmoid(value)
        calc_val = sig * (1 - sig)
        return calc_val

        
    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron with the
        given input. Same as g'(z) in book (note that this is not rounded.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The derivative of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        newActs = [1.0] + inActs
        weighted_sum = self.getWeightedSum(newActs)
        return self.sigmoidDeriv(weighted_sum)

    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
            alpha (float): The learning rate
            delta (float): If this is an output, then g'(z)*error
                           If this is a hidden unit, then the as defined-
                           g'(z)*sum over weight*delta for the next layer
        Returns:
            float
            Return the total modification of all the weights (sum of each abs(modification))
        """
        totalModification = 0
        """YOUR CODE"""
        newActs = [1.0] + inActs
        for (i, weight) in enumerate(self.weights):
            c_val = alpha * delta * newActs[i]
            self.weights[i] += c_val
            totalModification += abs(c_val)

        return totalModification
            
    def setRandomWeights(self):
        """
        Generates random input weights that vary from -1.0 to 1.0
        """
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))
        
    def __str__(self):
        """ toString """
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):                                    
    """
    Class to hold the net of perceptrons and implement functions for it.
    """          
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        """
        Initiates the NN with the given sizes.
        
        Args:
            layerSize (list<int>): the number of perceptrons in each layer 
        """
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1
        
        #build hidden layer(s)        
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h+1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)
 
        #build output layer
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)
            
        #build layers list that holds all layers in order - use this structure
        # to implement back propagation
        self.layers = [self.hiddenLayers[h] for h in xrange(self.numHiddenLayers)] + [self.outputLayer]
  
    def __str__(self):
        """toString"""
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr
    
    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.
        
        Args:
            inActs (list<float>): the input to the NN (an example) 
        Returns:
            list<list<float/int>>
            A list of lists. The first list is the input list, and the others are
            lists of the output values of all perceptrons in each layer.
        """
        """YOUR CODE"""
        curr_acts = inActs
        result = [inActs]

        for layer in self.layers:
            next_actions = []
            for n in layer:
                n_sigmoid = n.sigmoidActivation(curr_acts)
                next_actions.append(n_sigmoid)
            result.append(next_actions)
            curr_acts = next_actions

        return result
    
    def backPropLearning(self, examples, alpha):
        """
        Run a single iteration of backward propagation learning algorithm.
        See the text and slides for pseudo code.
        
        Args: 
            examples (list<tuple<list<float>,list<float>>>):
              for each tuple first element is input(feature)"vector" (list)
              second element is output "vector" (list)
            alpha (float): the alpha to training with
        Returns
           tuple<float,float>
           
           A tuple of averageError and averageWeightChange, to be used as stopping conditions. 
           averageError is the summed error^2/2 of all examples, divided by numExamples*numOutputs.
           averageWeightChange is the summed absolute weight change of all perceptrons, 
           divided by the sum of their input sizes (the average weight change for a single perceptron).
        """
        #keep track of output
        average_error = 0
        average_weight_change = 0
        n = 0
        
        for example in examples:
            deltas = []
            all_layer = self.feedForward(example[0])
            last_layer = all_layer[-1]
            out = []
            for outNum in xrange(len(example[1])):
                g_prime = self.outputLayer[outNum].sigmoidActivationDeriv(all_layer[-2])
                error = example[1][outNum] - last_layer[outNum]
                delta = error * g_prime
                average_error += error**2/2
                out.append(delta)
            deltas.append(out)
            
            """
            Backpropagate through all hidden layers, calculating and storing
            the deltas for each perceptron layer.
            """
            for lay in xrange(self.numHiddenLayers-1,-1,-1):
                layer = self.layers[lay]
                nextLayer = self.layers[lay + 1]
                hid_delt = []
                for num_neurons in xrange(len(layer)):
                    g_prime = layer[num_neurons].sigmoidActivationDeriv(all_layer[lay])
                    nex_err = 0
                    for nextNeuron in range(len(nextLayer)):
                        weight_delt = deltas[0][nextNeuron]
                        weight = nextLayer[nextNeuron].weights[num_neurons + 1]
                        deltaWeight = weight * weight_delt
                        nex_err += deltaWeight
                    delta = g_prime * nex_err
                    hid_delt.append(delta)
                deltas = [hid_delt] + deltas
            
            """
            Having aggregated all deltas, update the weights of the 
            hidden and output layers accordingly.
            """      
            for numLayer in xrange(0,self.numLayers):
                layer = self.layers[numLayer]
                for numNeuron in xrange(len(layer)):
                    weight_modification = layer[numNeuron].updateWeights(all_layer[numLayer], alpha, deltas[numLayer][numNeuron])
                    average_weight_change += weight_modification
                    n += layer[numNeuron].inSize
        tot_examples = len(examples) * len(examples[0][1])
        average_error /= tot_examples
        average_weight_change /= n
        return (average_error, average_weight_change)
    
def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008,hiddenLayerList = [1], maxItr = sys.maxint, startNNet = None):
    """
    Train a neural net for the given input.
    
    Args: 
        examples (tuple<list<tuple<list,list>>,
                        list<tuple<list,list>>>): A tuple of training and test examples
        alpha (float): the alpha to train with
        weightChangeThreshold (float):           The threshold to stop training at
        maxItr (int):                            Maximum number of iterations to run
        hiddenLayerList (list<int>):             The list of numbers of Perceptrons 
                                                 for the hidden layer(s). 
        startNNet (NeuralNet):                   A NeuralNet to train, or none if a new NeuralNet
                                                 can be trained from random weights.
    Returns
       tuple<NeuralNet,float>
       
       A tuple of the trained Neural Network and the accuracy that it achieved 
       once the weight modification reached the threshold, or the iteration 
       exceeds the maximum iteration.
    """
    examplesTrain,examplesTest = examples       
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])     
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print "Starting training at time %s with %d inputs, %d outputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)                                                    
    if startNNet is not None:
        nnet =startNNet
    """
    YOUR CODE
    """
    iteration = 0
    train_error = 0
    weight_modification = 0
    
    """
    Iterate for as long as it takes to reach weight modification threshold
    """
        
    weight_modification = weightChangeThreshold + 1
    while (iteration < maxItr and weight_modification > weightChangeThreshold):
        iteration += 1
        train_error, weight_modification = nnet.backPropLearning(examplesTrain, alpha)
          
    time = datetime.now().time()
    print 'Finished: %d iterations Time: %s Training Error: %f and Weight Change: %f' % (iteration,str(time),train_error,weight_modification)
                
    """
    Get the accuracy of your Neural Network on the test examples.
	For each text example, you should first feedforward to get the NN outputs. Then, round the list of outputs from the output layer of the neural net.
	If the entire rounded list from the NN matches with the known list from the test example, then add to testCorrect, else add to  testError.
    """ 
    
    test_error = 0
    test_correct = 0     
    
    for e in examplesTest:
        result = nnet.feedForward(e[0])
        r = []
        for i in range(len(result[-1])):
            res = round(result[-1][i])
            r.append(res)
        if r != e[1]:
            test_error += 1
        else:
            test_correct += 1
    

    test_combined = test_error + test_correct
    testAccuracy = float(test_correct) / test_combined
    
    print 'Feed Forward Test correctly classified: %d, incorrectly classified: %d, test percent error:  %f\n' % (test_correct , test_error, testAccuracy)
    
    """return something"""
    return nnet, testAccuracy



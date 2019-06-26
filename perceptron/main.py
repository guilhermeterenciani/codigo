#!/usr/bin/python
# coding: utf-8
import numpy as np
import time;
class NeuralNetwork(object):
    """docstring for ClassName"""
    def __init__(self, x,y):
        super(NeuralNetwork, self).__init__()
        self.inputs = np.array(x)
        self.y = np.array(y)

        self.weight1 = np.random.rand(self.inputs.shape[1],4);
        self.weight2 = np.random.rand(4,1);
        
        self.output = np.zeros(self.y.shape);

    
    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.inputs,self.weight1))
        self.layer2 = self.sigmoid(np.dot(self.layer1,self.weight2))
        return self.layer2;
    def sigmoid(self,x):
        return 1/(1+ np.exp(-x));
    
    def sigmoid_derivative(self,x):
        return x*(1-x)
    def train(self,x,y):
        self.output = self.feedforward();
        self.backpropagation()
    def backpropagation(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*self.sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.inputs.T, np.dot(2*(self.y -self.output)*self.sigmoid_derivative(self.output), self.weight2.T)*self.sigmoid_derivative(self.layer1))
        self.weight1 += d_weights1
        self.weight2 += d_weights2
    def predict(self,input):
        layer1 = self.sigmoid(np.dot(input,self.weight1))
        output = self.sigmoid(np.dot(layer1,self.weight2))
        return output;



if __name__ == "__main__":
    data = [[0,0,0,0],
            [0,0,1,1],
            [0,1,0,1],
            [0,1,1,1],
            [1,0,0,1],
            [1,0,1,1],
            [1,1,0,1],
            [1,1,1,0]];
    data = np.array((data),dtype=float);
    x = data[:,:-1];
    y =data[:,-1::1]
    obj = NeuralNetwork(x,y);
    for i in range(2000):
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(x))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(obj.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - obj.feedforward())))) # mean sum squared loss
        print ("\n")
        #time.sleep(0.2);
        obj.train(x,y);

    print(obj.predict([1,1,1]))
    print(obj.predict([0,0,0]))
    print(obj.predict([0,0,1]))

        
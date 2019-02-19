import numpy as np

#Transition matrix
T = np.array([[0.7, 0.3], 
              [0.3, 0.7]])

#Observation matrix
O = np.array([[[0.1, 0.0], 
               [0.0, 0.8]],
                
              [[0.9, 0.0], 
               [0.0, 0.2]]])

#Initial probability for rain
prob = np.array([0.5, 0.5])

#Observations of umbrella use
observations = [1, 1, 0, 1, 1]

def normalize(fm):
    return fm * (1 / np.sum(fm))

def forward(f, e):
    return normalize (O[e] @ T.T @ f)

def backward(b, e):
    return T @ O[e] b

#PART B ANSWER
for i in range(len(observations)):
    prob = forward(prob, observations[i])
    print('The probability for rain on day {} is: {}'.format(i + 1, prob[0]))




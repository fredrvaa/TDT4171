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

#Evidence of umbrella use
evidence = np.array([1, 1, 0, 1, 1])

#Normalizing function to normalize to 1
def normalize(m):
    return m * (1 / np.sum(m))

#Forward/filtering function
def forward(f, e):
    return normalize(O[e] @ T.T @ f)

#Backward/smoothing function
def backward(b, e):
    return T @ O[e] @ b

#ForwardBackward function
def forward_backward(ev, prior):
    fv = np.zeros((ev.shape[0] + 1, prior.shape[0]))
    b = np.ones(prior.shape[0])
    sv = np.zeros((ev.shape[0], prior.shape[0]))

    fv[0] = prior
    for i in range(1, ev.shape[0] + 1):
        fv[i] = forward(fv[i-1], ev[i-1])
    print('Backward messages:')
    for i in range(ev.shape[0]-1, -1, -1):
        print('b_{}: {}'.format(i+2, b))
        sv[i] = normalize(fv[i+1] * b)
        b = backward(b, ev[i])
    return sv


#Printing functions
def print_forward(f, e):
    fc = f.copy()
    for i in range(e.shape[0]):
        fc = forward(fc, e[i])
        print('The probability for rain on day {} is: {}'.format(i + 1, fc))

def print_forward_backward(ev, prior):
    sv = forward_backward(ev, prior)
    print('\n')
    for i in range(ev.shape[0]):
        print('The probability for rain on day {} is: {}'.format(i + 1, sv[i]))

#PART B
print('----PART B - Forward Algorithm----')
print('2 days:')
print_forward(prob, evidence[0:2])

print('\n 5 days:')
print_forward(prob, evidence) 


#Part C
print('\n ----PART C - ForwardBackward Algorithm----')
print('2 days:')
print_forward_backward(evidence[0:2], prob)

print('\n 5 days:')
print_forward_backward(evidence, prob)

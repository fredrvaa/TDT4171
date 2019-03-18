import numpy as np
import random
from utils import read_data

class Node(object):
    def __init__(self, value):
        self.value = value
        self.children = {}

def plurality_value(examples):
    p = 0
    n = 0
    for example in examples:
        if int(example[-1]) == 2:
            p += 1
        else:
            n += 1

    if p > n:
        return 2
    elif p < n:
        return 1
    return random.randint(1,2)

def same_class(examples):
    value = examples[0][-1]
    for example in examples:
        if value == example[-1]:
            return True
    return False

def B(q):
    if q == 0 or q == 1:
        return q
    else:
        return -(q*np.log(q, 2) + (1 - q)*np.log((1 - q), 2))

def entropy(value, attribute):
    num_positives = 0

    for v in value: 
        if v[attribute] == value[0][attribute]:
            num_positives += 1

    q = num_positives / len(value)
    return B(q)

def importance(value, attributes):
    if importance_type == 'random':
        return attributes[random.randint(0, len(attributes)-1)]
    elif importance_type == 'information_gain':
        entropies = {}
        for attribute in attributes:
            entropies[attribute] = entropy(value, attribute)

        minimum_value = 1
        a = 1
        for e in entropies:
            if entropies[e] < minimum_value:
                minimum_value = entropies[e]
                a = e
        return a                

def decision_tree_learning(examples, attributes, parent_examples):
    if len(examples) == 0: 
        print('a')
        return Node(plurality_value(parent_examples))
    elif same_class(examples):
        print('b')
        return same_class(examples)
    elif len(attributes) == 0:
        print('c')
        return Node(plurality_value(examples))
    else:
        A = importance(examples, attributes)
        tree = Node(A)
        
        for value in range(1,3):
            exs = []
            for example in examples:
                if int(example[A]) == value:
                    exs.append(example)
                children = decision_tree_learning(exs, attributes, examples)
                tree.children[value] = children
        return tree

def classify(node, example):
	while node.children:
		node = node.children[int(example[node.value])]
	return node.value

def testing(tree, value):
	matches = 0
	for example in value:
		if example[-1] == classify(tree, example):
			matches += 1
	print('Tests matching: {}'.format(matches))
    
#Defining types for importance algorithm
importance_types = ['random', 'information_gain']
importance_type = importance_types[0] #Set to 0/1

#Reading in train and test data
train = read_data('data/training.txt')
test = read_data('data/test.txt')

#Loading attributes

attributes = [x for x in range(len(train[0])-1)]

tree = decision_tree_learning(train, attributes, [])
print(tree)

testing(tree, test)


if '__name__' == '__main__':
    print('test')
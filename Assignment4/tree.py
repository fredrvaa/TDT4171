import math
import random
import numpy as np
from utils import read_data

class Node(object):
    def __init__(self, value, node_type):
        self.value = value
        self.node_type = node_type
        self.children = {}

def plurality_value(examples):
    p = 0
    n = 0
    for example in examples:
        if example[-1] == 2:
            p += 1
        else:
            n += 1

    if p > n:
        return 2
    elif p < n:
        return 1
    else:
        return random.randint(1,2)

def same_class(examples):
    value = examples[0][-1]
    for example in examples[1:]:
        if value != example[-1]:
            return False
    return True

def B(q):
    if q == 0 or q == 1:
        return q
    else:
        return -(q * math.log(q, 2) + (1 - q) * math.log((1 - q), 2))

def entropy(examples):
    num_positives = 0

    for example in examples:
        if example[-1] == 2:
            num_positives += 1
    print(len(examples))
    q = num_positives / len(examples) 

    return B(q)

def class_lists(attribute, examples):
    c1 = []
    c2 = []
    for example in examples:
        if example[attribute] == 1:
            c1.append(example)
        else:
            c2.append(example)

    return c1, c2

def find_remainder(c1, c2, tot):
    return len(c1) / tot * entropy(c1) + \
           len(c2) / tot * entropy(c2)

def importance(attribute, examples, importance_type):
    if importance_type == 'random':
        return attributes[random.randint(0, len(attributes)-1)]
    elif importance_type == 'information_gain':
        goal = entropy(examples)

        c1, c2 = class_lists(attribute, examples)
        remainder = find_remainder(c1, c2, len(examples))

        return goal - remainder             

def decision_tree_learning(examples, attributes, parent_examples, importance_type):
    if len(examples) == 0: 
        return Node(plurality_value(parent_examples), 'leaf')
    elif same_class(examples):
        return Node(examples[0][-1], 'leaf')
    elif len(attributes) == 0:
        return Node(plurality_value(examples), 'leaf')
    else:
        importance_list = []

        for attribute in attributes:
            importance_list.append(importance(attribute, examples, importance_type))

        A = attributes[np.argmax(importance_list)]

        tree = Node(A, 'root')

        other_attributes = list(attributes)
        other_attributes.remove(A)

        class_examples = {1:[], 2:[]}

        for example in examples:
            class_examples[example[A]].append(example)
    
        for v, exs in class_examples.items():
            tree.children[v] = decision_tree_learning(exs, other_attributes, examples, importance_type)

        return tree

def classify(node, example):
	while node.children:
		node = node.children[int(example[node.value])]

	return node.value

def number_of_matches(tree, examples):
    matches = 0
    for example in examples:
        if example[-1] == classify(tree, example):
            matches += 1

    return matches

def testing(num_tests, importance_type, train, test, attributes):
    matches = 0
    for i in range(num_tests):
        tree = decision_tree_learning(train, attributes, [], importance_type)
        matches += number_of_matches(tree, test)
    matches /= num_tests
    print('Average of {}/{} matches after {} tests using {} importance'.format(matches, len(train), num_tests, importance_type))

if __name__ == '__main__':
    #Reading in train and test data
    train = read_data('data/training.txt')
    test = read_data('data/test.txt')

    #Loading attributes
    attributes = [x for x in range(len(train[0])-1)]

    #Testing random importance
    testing(1, 'random', train, test, attributes)
    testing(100, 'random', train, test, attributes)

    #Testing information gain importance
    testing(1, 'information_gain', train, test, attributes)
    testing(100, 'information_gain', train, test, attributes)

import math
import random
import numpy as np
from TreeNode import TreeNode

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
    q = num_positives / len(examples) 

    return B(q)

def class_lists(attribute, examples):
    c1 = []
    c2 = []
    for example in examples:
        if example[attribute] == 2:
            c2.append(example)
        else:
            c1.append(example)

    return c1, c2

def find_remainder(c1, c2, tot):
    return len(c1) / tot * entropy(c1) + \
           len(c2) / tot * entropy(c2)

def importance(attributes, attribute, examples, importance_type):
    if importance_type == 'random':
        return attributes[random.randint(0, len(attributes)-1)]
    elif importance_type == 'information_gain':
        goal = entropy(examples)

        c1, c2 = class_lists(attribute, examples)
        remainder = find_remainder(c1, c2, len(examples))

        return goal - remainder    

def same_class(examples):
    value = examples[0][-1]
    for example in examples[1:]:
        if value != example[-1]:
            return False
    return True
         

def decision_tree_learning(examples, attributes, parent_examples, importance_type):
    if len(examples) == 0: 
        return TreeNode(plurality_value(parent_examples), 'leaf')
    elif same_class(examples):
        return TreeNode(examples[0][-1], 'leaf')
    elif len(attributes) == 0:
        return TreeNode(plurality_value(examples), 'leaf')
    else:
        importance_list = []

        for attribute in attributes:
            importance_list.append(importance(attributes, attribute, examples, importance_type))
        
        A = attributes[np.argmax(importance_list)]

        class_examples = {1:[], 2:[]}

        for example in examples:
            class_examples[example[A]].append(example)

        tree = TreeNode(A, 'root')

        other_attributes = list(attributes)
        other_attributes.remove(A)

        for v, exs in class_examples.items():
            tree.children[v] = decision_tree_learning(exs, other_attributes, examples, importance_type)

        return tree


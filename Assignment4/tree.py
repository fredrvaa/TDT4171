import numpy as np

#Defining types for importance algorithm
importance_types = ['random', 'information_gain']
importance_type = importance_types[0] #Set to 0/1

class node(object):
    def __init__(self, value, ntype):
        self.value = value
        self.ntype =  ntype
        self.branches = {}

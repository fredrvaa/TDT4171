from utils import read_data
from test import testing
from TreeNode import TreeNode, print_tree

if __name__ == '__main__':
    #Reading in train and test data
    train = read_data('data/training.txt')
    test = read_data('data/test.txt')

    #Loading attributes
    attributes = [x for x in range(len(train[0])-1)]

    #Testing random importance
    tree_r = testing(1, 'random', train, test, attributes)
    testing(100, 'random', train, test, attributes)

    #Testing information gain importance
    tree_ig = testing(1, 'information_gain', train, test, attributes)
    testing(100, 'information_gain', train, test, attributes)

    #Printing trees using random- and information gain importance respectively
    print('TREE STRUCTURE USING RANDOM IMPORTANCE')
    print_tree(tree_r)
    print('TREE STRUCTURE USING INFORMATION GAIN IMPORTANCE')
    print_tree(tree_ig)
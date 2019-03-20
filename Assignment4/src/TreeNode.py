class TreeNode(object):
    def __init__(self, value, node_type):
        self.children = {}
        self.value = value
        self.node_type = node_type

def print_tree(node, t=0):
    tab = t+1
    if node.node_type == 'leaf':
        tree = 'Class ' + str(node.value) + '\n'
    else:
        tree = 'Attribute ' + str(node.value + 1) + '\n'
    for child in node.children:
        tree = tree +  t * '\t'
        if type(node.children[child]) == TreeNode:
            tree = tree + str(print_tree(node.children[child], tab))
        else:
            tree = tree + str(node.children[child]) + '\n'
    if tab != 1:
        return tree
    else:
        print(tree)
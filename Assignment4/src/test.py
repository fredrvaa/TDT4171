from decision_tree_learning import decision_tree_learning

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
    m_28 = 0
    for i in range(num_tests):
        tree = decision_tree_learning(train, attributes, [], importance_type)
        matches += number_of_matches(tree, test)
        if(number_of_matches(tree, test)==28):
            m_28 += 1
    matches /= num_tests
    print('Average of {}/{} matches after {} tests using {} importance. Total of {} perfect tests.'.format(matches, len(train), num_tests, importance_type, m_28))
    return tree
import math

# Creating a Node in the decision tree
class Node:
    def __init__(self, data=None, label=None, leaf=False):
        self.data = data
        self.label = label
        self.leaf = leaf
        self.children = {}

# Function to build a ID3 decision tree
def build_tree(data, features):

    labels = [row[-1] for row in data]
    
    # check if all examples have the same label
    if len(set(labels)) == 1:
        return Node(label=labels[0], leaf=True)
    
    # check if there are no more features to split on
    if not features:
        return Node(label=max(set(labels), key=labels.count), leaf=True)
    
    # find the best feature to split on using information gain
    max_gain = -1
    best_feature = None
    for feature in features:
        gain = information_gain(data, feature)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature
    
    # create a new decision node with the best feature
    root = Node(label=best_feature)
    feature_values = []
    for row in data:
        if row[best_feature] not in feature_values:
            feature_values.append(row[best_feature])
    feature_values = set(feature_values)
    
    # recursively build the subtree using the remaining features
    for value in feature_values:
        # create a subset of the data where the best feature has the given value
        subset = []
        for row in data:
            if row[best_feature] == value:
                subset.append(row)
        if len(subset) == 0:
            # if the subset is empty, create a leaf node with the most common label
            root.children[value] = Node(label=max(set(labels), key=labels.count), leaf=True)
        else:
            # otherwise, create a child node and recurse
            remaining_features = []
            for f in features:
                if f != best_feature:
                    remaining_features.append(f)
            root.children[value] = build_tree(subset, remaining_features)
    
    return root

def information_gain(data, feature):
    # calculate the entropy of the whole dataset
    label_counts = {}
    for row in data:
        label = row[-1]
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    total_entropy = entropy([count / len(data) for count in label_counts.values()])
    
    # calculate the entropy of the dataset for each feature value
    feature_values = set([row[feature] for row in data])
    subset_entropies = []
    subset_sizes = []
    for value in feature_values:
        subset = []
        for row in data:
            if row[feature] == value:
                subset.append(row)
        subset_sizes.append(len(subset))
        subset_label_counts = {}
        for row in subset:
            label = row[-1]
            if label in subset_label_counts:
                subset_label_counts[label] += 1
            else:
                subset_label_counts[label] = 1
        subset_entropy = entropy([count / len(subset) for count in subset_label_counts.values()])
        subset_entropies.append(subset_entropy)
    
    # calculate the weighted average of the subset entropies
    weighted_entropy = 0
    for i in range(len(subset_sizes)):
        weighted_entropy += (subset_sizes[i] / len(data)) * subset_entropies[i]

    # return the information gain
    info_gain = total_entropy - weighted_entropy

    return info_gain

# Function to calculate the entropy for a list of probabilities
def entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p != 0:
            entropy -= p * math.log2(p)
    return entropy

# Function to predict a class, given the decision tree
def predict(tree, attributes):
    if tree.leaf:
        return tree.label
    else:
        attribute_value = attributes[tree.label]
        if attribute_value not in tree.children:
            return "-1"
        else:
            child = tree.children[attribute_value]
            return predict(child, attributes)
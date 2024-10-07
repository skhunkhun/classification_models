from kMeans import *
from decisionTree import *
import csv

data = []
with open('iris.data') as f:
    reader = csv.reader(f)
    # Convert the numeric values to floats and store them as tuples in a list
    for row in reader:
        if len(row) != 0:
            tup = (float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4])
            data.append(tup)

random.shuffle(data)
test_size = 30

test_data = data[:test_size]
train_data = data[test_size:]

clusters = kmeans(train_data, 3, 20)
prediction = predict_classes(test_data, clusters)

total_setosa = 0
total_versi = 0 
total_virginica = 0

for i, rand in enumerate(test_data):
    if 'setosa' in rand[4]:
        total_setosa += 1
    elif 'versicolor' in rand[4]:
        total_versi += 1
    else:
        total_virginica += 1

num_setosa = 0
num_versi = 0
num_virginica = 0

for item in prediction:
    if 'setosa' in item:
        num_setosa += 1
    elif 'versicolor' in item:
        num_versi += 1
    else:
        num_virginica += 1

temp_ver_total = total_versi
temp_set_total = total_setosa
temp_vir_total = total_virginica

if num_versi > total_versi:
    temp = num_versi
    num_versi = temp_ver_total
    temp_ver_total = temp

if num_virginica > total_virginica:
    temp = num_virginica
    num_virginica = temp_vir_total
    temp_vir_total = temp

if num_setosa > total_setosa:
    temp = num_setosa
    num_setosa = temp_set_total
    temp_set_total = temp

print("K-Means:")
print(f"Iris Setosa accuracy: {round((num_setosa / temp_set_total) * 100, 0)}%")
print(f"Iris Versicolour accuracy: {round((num_versi / temp_ver_total) * 100, 0)}%")
print(f"Iris Virginica accuracy: {round((num_virginica / temp_vir_total) * 100, 0)}%")

features = [0, 1, 2, 3]
decision_tree = build_tree(train_data, features)

decision_setosa = 0
decision_versi = 0
decision_virginica = 0

for flower in test_data:
    predict_class = predict(decision_tree, flower)
    if 'setosa' in predict_class:
        if predict_class == flower[4]:
            decision_setosa += 1
    elif 'versicolor' in predict_class:
        if predict_class == flower[4]:
            decision_versi += 1
    elif 'virginica' in predict_class:
        if predict_class == flower[4]:
            decision_virginica += 1

if decision_versi > total_versi:
    temp = decision_versi
    decision_versi = total_versi
    total_versi = temp

if decision_virginica > total_virginica:
    temp = decision_virginica
    decision_virginica = total_virginica
    total_virginica = temp

if decision_setosa > total_setosa:
    temp = decision_setosa
    decision_setosa = total_setosa
    total_setosa = temp

print("\nDecision Tree:")
print(f"Iris Setosa accuracy: {round((decision_setosa / total_setosa) * 100, 0)}%")
print(f"Iris Versicolour accuracy: {round((decision_versi / total_versi) * 100, 0)}%")
print(f"Iris Virginica accuracy: {round((decision_virginica / total_virginica) * 100, 0)}%")
from trainheartattack import DNA
import numpy as np

with open("normdata.csv", "r") as f:
    X = []
    Y = []
    for line in f.readlines():
        data = list(map(float, line.strip().split(",")))
        X.append(data[:6])  # put inputs in an array
        Y.append(data[-1])  # put outputs in an array
    Y = np.array(Y)  # main outputs from csv
    X = np.array(X)  # main inputs from csv
    # print(Y)

# print(Y.shape) total of 8763 data from csv
Xtrain = X[:5000]
Ytrain = Y[:5000]
Xtest = X[5000:]
Ytest = Y[5000:]

dna = DNA()
with open('newWeights.txt', 'r') as f:  # open and read data
    for y in range(6):
        for x in range(3):
            dna.weights[y][x] = float(f.readline())
# print(dna.weights)

with open('newWeights2.txt', 'r') as f:  # open and read data
    for x in range(3):
        dna.weights2[x] = float(f.readline())
# print(dna.weights2)


errors = []
pred = []
for i in range(len(Xtest)):
    inputs = Xtest[i]
    predicted_value = dna.forward(inputs)
    pred.append(predicted_value)
    true_value = Ytest[i]
    errors.append(abs(true_value-predicted_value))

max=np.max(pred)
min=np.min(pred)
mid=(max+min)/2
print('Minimum Value:',min)
print('Maximum Value:',max)
print('Midpoint:',mid)
# pred = (np.mean(pred))
# print('Predicted value:', pred)
errors = (np.mean(errors))
print('Error: ', errors)

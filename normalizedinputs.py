import numpy as np
with open("Copy of heart_attack_prediction_dataset(1).csv", "r", encoding="utf8") as f:
    f.readline()  # buang first line
    data = []
    heart = []
    for line in f.readlines():
        line = line.replace("Male", "1")
        line = line.replace("Female", "0")
        inp = line.split(",")[1:4]+line.split(",")[5:7]+line.split(",")[15:16]
        inp = list(map(float, inp))  # change str to float
        heart.append(line.split(",")[-1])
        data.append(inp)  # add to array

#print(len(data))  # length
data = np.array(data)
#print(data.shape)

m = np.mean(data, axis=0)  # mean
#print(list(m))
s = np.std(data, axis=0)  # std dev
#print(list(s))

#print('Data 0', data[0])
data = (data-m)/s  # norm
#print('Data 0', data[0])

with open("normdata.csv", "w+") as f:
    for row, output in zip(data, heart):
        row = list(map(str, row))
        line = ','.join(row) + ',' + output
        f.write(line)

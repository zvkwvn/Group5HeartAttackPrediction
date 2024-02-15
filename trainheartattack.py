import numpy as np

np.random.seed(0)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def normalize(x):
    return (x-np.mean(x))/np.std(x)


def mse(x):
    #return np.sqrt(np.sum(np.power(x, 2))/len(x))
    return (np.sum(np.power(x, 2)/2)/len(x))
    # return (np.mean(np.power(x, 2)/2))

# X = np.array([[1, 2, 3, 4],
#               [5, 6, 7, 8]])
# weights = [[1, 2, 3, 4],
#            [5, 6, 7, 8],
#            [9, 1, 2, 3]]
# bias1 = [1, 2, 3]

# hidden1 = relu(np.dot(X, np.array(weights).T)+bias1)


with open("normdata.csv", "r") as f:
    X = []  # input of data
    Y = []  # output of data
    for line in f.readlines():
        data = list(map(float, line.strip().split(",")))
        X.append(data[:6])  # put inputs in an array
        Y.append(data[-1])  # put outputs in an array
    Y = np.array(Y)  # main outputs from csv
    X = np.array(X)  # main inputs from csv
    # print(Y)

# print(Y.shape) total of 8763 data from csv
Xtrain = X[:5000]  # data for training
Ytrain = Y[:5000]
Xtest = X[5000:]  # data for testing
Ytest = Y[5000:]

# weights = np.random.randn(6, 3)  # generate weights in 6x3
# hin = np.dot(X[0:100], weights)
# hout = sigmoid(hin)

# weights2 = np.random.randn(3, 1)
# oin = np.dot(hout, weights2)
# oout = sigmoid(oin)
# # print(oout)


class DNA:  # every one is a Neural Network
    def __init__(self):
        # self. is for storing DNA's weight
        self.weights = np.random.randn(6, 3)
        self.weights2 = np.random.randn(3, 1)

    def forward(self, input):  # forward propagation
        hin = np.dot(input, self.weights)
        hout = sigmoid(hin)
        oin = np.dot(hout, self.weights2)
        oout = sigmoid(oin)
        return oout

    def mutate(self, mutation_rate):  # mutation (to exit local minima, introduce variation)
        for i, weight in enumerate(self.weights): #every weight will be add or
            amount = np.mean(weight) * mutation_rate #subtract a certain amount
            add_random = np.vectorize(
                lambda x: x+np.random.uniform(-amount, amount))
            self.weights[i] = add_random(weight)
        for i, weight2 in enumerate(self.weights2):
            amount = np.mean(weight2) * mutation_rate
            add_random = np.vectorize(
                lambda x: x+np.random.uniform(-amount, amount))
            self.weights2[i] = add_random(weight2)


dna = DNA()
if __name__ == '__main__':
    population: list[DNA] = []
    pop = 100
    batch_size = 32
    mutation_rate = 0.1
    generation = 0
    for i in range(pop):
        population.append(DNA())
    try:
        while True:
            generation += 1
            fitnesses = []
            for dna in population:
                errors = []
                for x in range(batch_size):
                    # randomly pick any row in csv
                    random_index = np.random.randint(0, 5000)
                    inputs = Xtrain[random_index]
                    predicted_value = dna.forward(inputs)  # forward pass NN
                    true_value = Ytrain[random_index]
                    errors.append(abs(true_value-predicted_value))
                fitness = 1/mse(errors)
                fitnesses.append(fitness)
            total_fitness = np.sum(fitnesses)
            new_population = []
            for _ in range(pop):
                chosen = np.random.uniform(0, total_fitness)
                cumulative_fitness = 0 #fittest will get mutated
                # store its position number in array
                for i, fitness in enumerate(fitnesses):
                    cumulative_fitness += fitness
                    if cumulative_fitness > chosen:  # to find if the roulette hits
                        chosen_dna = population[i]
                        chosen_dna.mutate(mutation_rate)
                        break
                new_dna = DNA()  # making new dna
                new_dna.weights = chosen_dna.weights.copy()
                new_dna.weights2 = chosen_dna.weights2.copy()
                new_population.append(new_dna)
            population = new_population.copy()
            print(f'Generation: {generation:4.0f}',
                  f'Fitness: {np.mean(fitnesses):7.5f}')
    except KeyboardInterrupt:
        print('Training stopped.')
        with open("newWeights.txt", "w+") as f:  # save and write the new weights
            best_index = np.argmax(fitnesses)  # find max index for best dna
            best_dna = population[best_index]
            best_dna.weights
            for row in best_dna.weights:
                # print(row)
                for line in row:
                    # print(line)
                    f.write(str(line) + '\n')
        with open("newWeights2.txt", "w+") as f:  # save and write the new weights
            best_index = np.argmax(fitnesses)  # find max index for best dna
            best_dna = population[best_index]
            best_dna.weights2
            for row in best_dna.weights2:
                # print(row)
                for line in row:
                    # print(line)
                    f.write(str(line) + '\n')

            # row = list(map(str,row))
            # line=','.join(row) + ',' + output
            # f.write(line)

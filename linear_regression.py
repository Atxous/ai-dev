import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ai_Classes as ai

np.random.seed(0)

def standardize(x):
    mean = x - np.mean(x, axis=0)
    x = mean / np.std(x, axis = 0)
    return x

def normalize(x, min = 0):
    x = (x - x.min(axis = 0)) / (x.max(axis = 0) - x.min(axis = 0))
    if min == -1:
        x = (2 * x) - 1
    return x

'''
x = np.random.uniform(low = -10, high = 20, size = [4, 3])
print(x)
print(x.min(axis = 0), x.max(axis = 0))

x = normalize(x)
print("\n", x)
print(np.mean(x, axis = 0), np.std(x, axis = 0))
print(x.min(axis = 0), x.max(axis = 0))
'''

dataset = pd.read_csv("255585")
dataset = dataset.drop(labels = ["STATION", "NAME", "DATE"], axis = 1)
dataset = dataset.to_numpy()

moving_avg = 125
modified = np.empty([len(dataset) - moving_avg + 1, 2])
for i in range(len(modified)):
    modified[i] = np.mean(dataset[i:i+125], axis = 0)

x = np.array(range(len(modified)))
y = modified.copy()

#adding in x powers
degree = 3
xs = np.empty_like(x, shape = [len(x), degree])
for power in range(1, degree + 1):
    xs[:, power - 1] = np.power(x, power)
x = xs.copy()

x = normalize(x, min = -1)

model = ai.dense(3, 2)
mse = ai.costMSE()
optimizer = ai.SGD_Optimizer(0.1, mu = 0.9)

model.forward(x)
print("Initial Cost: ", mse.forward(model.outputs, y))

for epochs in range(200):
    model.forward(x)
    cost = mse.forward(model.outputs, y)
    if epochs % 20 == 0:
        print(cost)
    
    mse.backward(model.outputs, y)
    model.backward(mse.dinputs)
    optimizer.update_params(model)

model.forward(x)
cost = mse.forward(model.outputs, y)
print("Final Cost: ", cost)

fig = plt.figure()

plt.plot(y[:,0], label = "TMAX")
plt.plot(y[:,1], label = "TMIN")

plt.plot(model.outputs[:,0], label = "Prediction -- TMAX")
plt.plot(model.outputs[:,1], label = "Prediction -- TMIN")

plt.xlabel("Modified, Days of Year")
plt.ylabel("Temp (Celsius)")
plt.legend()

plt.show()
plt.close()

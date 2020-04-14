# %%
import numpy as np
import matplotlib.pyplot as plt
from random import seed, random, uniform, sample
from math import exp
from itertools import product

seed(6)

# %%

def generate_data (N):

    h_x = []
    samples = []

    for i in range(N):
        x_i = uniform(0.0, 1.0)
        n_i = uniform(-0.1, 0.1)
        h_xi = 0.5 + 0.4*np.sin(3*np.pi*x_i)

        samples.append((x_i, h_xi + n_i))
        h_x.append((x_i, h_xi))

    return samples, h_x


# %%

class Cluster:
    def __init__(self, center):
        self.var = 0.0
        self.center = center
        self.samples = []

    def getGaussian(self, x):
        return exp( -0.5/self.var*(x-self.center)**2 )

    def updateCenter(self):
        if len(self.samples) > 0:
            center_new = sum(self.samples) / len(self.samples)

            if self.center == center_new:
                converged = True
            else:
                converged = False
                self.samples = []
            self.center = center_new
            return converged
        else:
            return True

    def updateVariance(self, var = 0.0):
        if len(self.samples) <= 1:
            self.var = var
        else:
            self.var = np.var(self.samples)

# %%
class KmeansClustering:
    def __init__(self, K = 3):
        self.K = K
        self.clusters = []

    def fit (self, data, useSameVar):
        initCentroids = sample(data, self.K)

        for i in range(self.K):
            cluster = Cluster(initCentroids[i])
            self.clusters.append(cluster)

        converged = False
        while not converged:
            for x in data:
                dist = []
                for C in self.clusters:
                    dist.append( (x - C.center)**2 ) 
                # Find closest cluster centroid --> Add new point
                self.clusters[dist.index( min(dist) )].samples.append(x)

            converged = True
            for C in self.clusters:
                converged = converged and C.updateCenter()

            if not converged:
                for C in self.clusters:
                    C.samples = []

        if useSameVar: # Adjusting variance values --> Assign the mean if only one data
            cluster_dist = []
            for C_i in self.clusters:
                for C_j in self.clusters:
                    cluster_dist.append( (C_i.center- C_j.center)**2 )
            dMax = max( cluster_dist )
            dMax = dMax / (2.0 * self.K)
            for C in self.clusters:
                C.var = dMax

        else: # Same variance
            var_mean = 0.0
            count = 0
            for C in self.clusters:
                if len(C.samples) > 1:
                    C.updateVariance()
                    var_mean += C.var
                    count += 1
            var_mean = var_mean / count

            for C in self.clusters:
                if len(C.samples) <= 1:
                    C.updateVariance(var_mean)

        return self.clusters

# %%
class RBF_Regressor:

    def __init__(self, K = 3, eta = 0.01, useSameVar = True):
        self.K = K
        self.eta = eta
        self.useSameVar = useSameVar

        self.bias = uniform(0.0, 0.5)

        self.weights = []
        for i in range(self.K):
            self.weights.append(uniform(0.0, 0.5))


    def fit (self, samples):
        X = [x for x, _ in samples]

        kmeans = KmeansClustering(self.K)
        self.bases = kmeans.fit(X, self.useSameVar)

        for ep in range(100): 
            for x, d in samples:
                # Find the Loss
                y = 0.0
                for i in range(len(self.bases)):
                    y += self.bases[i].getGaussian(x) * self.weights[i]
                y += self.bias
               
                err = (d - y)

                # LMS Update Rule
                for j in range(self.K):
                    g = err * self.bases[j].getGaussian(x)
                    self.weights[j] += self.eta * g
                self.bias += self.eta * err

    def predict (self, X):
        y = []
        for x_i in X:
            y_i = 0.0
            for j in range( len(self.bases) ):
                y_i += self.bases[j].getGaussian(x_i) * self.weights[j]
            y_i +=  self.bias
            y.append(y_i)
        return y

# %%

K = [3, 6, 8, 12, 16] 
eta = [0.01, 0.02]
useSameVar = True 

samples, h_x = generate_data(75)
expParam = list(product(K, eta)) 

for K, eta in expParam:
    rbfNet = RBF_Regressor(K, eta, useSameVar)
    rbfNet.fit(samples)

    xx = np.linspace(0,1,200)
    y_predict = rbfNet.predict(xx.tolist())
    
    X = [s[0] for s in samples]
    y_true = [s[1] for s in samples]

    plt.figure()
    plt.scatter(X, y_true,  label="Training Samples", marker='o', facecolors='none', edgecolors='red')
    plt.plot(xx, 0.5 + 0.4*np.sin(3*np.pi*xx), 'g-', label='Original Function')
    plt.plot(xx, y_predict, '--', label='RBF Estimation')

    plt.axis([0.0, 1.0, -0.1, 1.1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$h(x)$")
    plt.title("$ K =  $" + str(K) + r", $ \eta =  $" + str(eta) + " with same variance")
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':')
plt.show()



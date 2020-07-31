import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9,11]])

new_data = np.array([[1,3],
                     [8,9],
                     [0,3],
                     [5,4],
                     [6,4]])

colors = ["g","r","c","b","k"] # if k>5 , replace this to colors = 10*["g","r","c","b","k"]

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        classification = []

        for dado in data:
            distances = [np.linalg.norm(dado-self.centroids[centroid]) for centroid in self.centroids]
            classification.append(distances.index(min(distances)))
            plt.scatter(dado[0], dado[1], marker="*", color=colors[classification[len(classification)-1]], s=50, linewidths=5)

        return classification
    
    def plot(self):

        for centroid in self.centroids:
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1], marker="o", color="k", s=5, linewidths=5)

        for classification in self.classifications:  
            color = colors[classification]
            for featureset in self.classifications[classification]:
                plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=100, linewidths=5)


clf = K_Means()
clf.fit(X)
clf.predict(new_data)
clf.plot()
plt.show()
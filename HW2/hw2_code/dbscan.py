import numpy as np
from kmeans import pairwise_dist
class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        C = 0
        vs = set()
        cluster_idx = np.ones(len(self.dataset))*-8
        # for each unvisited point P in dataset X
        for i in range (0, len(self.dataset)):
            if i not in vs:
                #     mark P as visited
                vs.add(i)
                #     NeighborPts = regionQuery(P, eps)
                neighborPts = self.regionQuery(i)
        #     if sizeof(NeighborPts) < MinPts
                if len(neighborPts) < self.minPts:
                    #     mark P as NOISE
                    cluster_idx[i] = -8
                else:
                    #     C = next cluster
                    self.expandCluster(i, neighborPts, C, cluster_idx, vs)
                    #     expandCluster(P, NeighborPts, C, eps, MinPts)
                    C = C + 1
        return cluster_idx

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints:  
            1. np.concatenate(), and np.sort() may be helpful here. A while loop may be better than a for loop.
            2. Use, np.unique(), np.take() to ensure that you don't re-explore the same Indices. This way we avoid redundancy.
        """
        #     add P to cluster C
        cluster_idx[index] = C
        #     for each point P' in NeighborPts
        i = 0
        while i < len(neighborIndices):
        #         if P' is not visited
            if neighborIndices[i] not in visitedIndices:
        #             mark P' as visited
                visitedIndices.add(neighborIndices[i])
        #             NeighborPts' = regionQuery(P', eps)
                neighborPts = self.regionQuery(neighborIndices[i])
        #             if sizeof(NeighborPts') >= MinPts
                if len(neighborIndices) >= self.minPts:
        #                 NeighborPts = NeighborPts joined with NeighborPts‘
                    neighborIndices = np.concatenate((neighborIndices, neighborPts))
        #         if P' is not yet member of any cluster
            if cluster_idx[neighborIndices[i]] < 0:
        #             add P' to cluster C
                cluster_idx[neighborIndices[i]] = C
            i = i + 1
        #     regionQuery(P, eps) return all points within P's eps-neighborhood (including P)

        return
    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        distances = pairwise_dist(self.dataset[pointIndex].reshape(1, len(self.dataset[0])), self.dataset)
        # print(distances)
        indices = np.argwhere(distances <= self.eps)
        # print(indices[:,1])
        return indices[:,1]
        
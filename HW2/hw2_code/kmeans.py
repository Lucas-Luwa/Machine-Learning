
'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        # print("X", len(self.points))
        # print(np.random.choice(len(self.points), self.K, False))
        # print("RUNNING ICenter")
        # print(self.points[np.random.choice(len(self.points), self.K, False)])
        return self.points[np.random.choice(len(self.points), self.K, False)]

    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """
        # 1. Sample 1% of the points from the dataset, uniformly at random (UAR) and without replacement. This sample will be the dataset t
        # he remainder of the algorithm uses to minimize initialization overhead.
        ptsPicked = self.points[np.random.choice(len(self.points), int(len(self.points)*.01), False)]
        # 2. From the above sample, select only one random point to be the first cluster center. 
        selectedPTs = ptsPicked[np.random.choice(len(ptsPicked), 1)]
        # 3. For each point in the sampled dataset, find the nearest cluster center and record the squared distance to get there.
        for i in range(0, self.K - 1):
        # 4. Examine all the squared distances and take the point with the maximum squared distance as a new cluster center. 
            sqrtDist = pairwise_dist(ptsPicked, selectedPTs)**2
        # In other words, 
        # we will choose the next center based on the maximum of the minimum calculated distance instead of sampling randomly like in step 2. 
            maxofMin = np.max(np.min(sqrtDist))
            # print(maxofMin)
            updatedCent = ptsPicked[(np.where(maxofMin == sqrtDist)[1][0])]
            # print(np.where(maxofMin == sqrtDist)[1][0])
            # print(updatedCent)
            # print(ptsPicked)
            selectedPTs = np.append(selectedPTs,[updatedCent], axis = 0)
            # print(selecstedPTs)
        # 5. Repeat 3-4 until all k-centers have been assigned. You may use a loop over K to keep track of the data in each cluster.
        # print(selectedPTs)
        # print(selectedPTs.shape[0], selectedPTs.shape[1]) # too many iterations by 1
        return selectedPTs

        # raise NotImplementedError

    def update_assignment(self):  # [5 pts]
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison. 
        """        
        # dist = self.pairwise_dist(self.centers, self.points)
        # print(pairwise_dist(self.centers, self.points))
        self.assignments = np.argmin(pairwise_dist(self.centers, self.points), axis=0)
        return self.assignments
        # raise NotImplementedError

    def update_centers(self):  # [5 pts]
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """
        retV = np.zeros(self.centers.shape)
        for i in range (0, self.K):
            retV[i] = np.mean(self.points[self.assignments== i],axis = 0)
        # print(retV)
        return retV

    def get_loss(self):  # [5 pts]
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """

        centerGroup = self.assignments
        retV = 0.0
        for i in range (0, len(self.centers)):
            currPts = self.points[centerGroup == i]
            dist = (currPts - self.centers[i])
            # print(currPts)
            # print(self.centers[i])
            retV += np.sum(dist ** 2)
        # print(retV)
        return retV

    def train(self):    # [10 pts]
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__ OK
                1. Update the cluster assignment for each point DONE
                2. Update the cluster centers based on the new assignments from Step 1 DONE

                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.

                4. Calculate the loss and check if the model has converged to break the loop early. DONE
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.   
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping! K
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """
        # print(self.assignments)
        oGLoss = 12823
        # print("init")
        for i in range (0, self.max_iters): # limit based on def
            self.assignments = self.update_assignment()
            self.centers = self.update_centers()        
            # print("we at 2")
            if self.K != len(np.unique(self.assignments)):
                # print("pass condition")
                noCluster = np.setdiff1d(np.arange(self.K), np.unique(self.assignments))
                # print(noCluster)
                for element in noCluster:
                    # print(element) 
                    self.centers[element] = self.points[np.random.choice(self.points.shape[0], 1, False)] #Note: Can also use len(self.points)
            self.loss = self.get_loss()
            if (np.abs(oGLoss - self.loss)/oGLoss) < self.rel_tol: break
            oGLoss = self.loss
        return self.centers, self.assignments, self.loss
        # raise NotImplementedError


def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """
        # print(x)
        # print(y)
        # raise NotImplementedError
        return np.sqrt(np.maximum(np.sum(x ** 2, axis = 1, keepdims = True) + np.sum(y ** 2, axis = 1) - 2 * np.dot(x, y.T), 0))

def rand_statistic(xGroundTruth, xPredicted): # [5 pts]
    """
    Args:
        xPredicted : N x 1 numpy array, N = no. of test samples
        xGroundTruth: N x 1 numpy array, N = no. of test samples
    Return:
        Rand statistic value: final coefficient value as a float
    """
    cmp_matrix1 = (np.array(xGroundTruth).reshape(-1, 1) == np.array(xGroundTruth))
    cmp_matrix2 = (np.array(xPredicted).reshape(-1, 1) == np.array(xPredicted))
    result = np.logical_not(np.logical_xor(cmp_matrix1, cmp_matrix2)).astype(int)# XNOR AKA not XOR
    # print(result)
    # print(cmp_matrix1)
    # print(cmp_matrix2)
    sumUpperTriangular = np.sum(np.triu(result, k=1)) #We don't want main diagonal
    # print(sumUpperTriangular)
    denominator = (len(xPredicted)*(len(xPredicted)-1))/2#((N)(N+1))/2 we sub in (N - 1)
    # print(denominator)
    # print(cmp_matrix)
    return float(sumUpperTriangular/denominator)
    # diagsum = np.trace(cmp_matrix1)  # Sum of diagonal elements
    # totsum = np.sum(cmp_matrix1) - diagsum
    # print(diagsum, totsum)
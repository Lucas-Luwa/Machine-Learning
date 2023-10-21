'''
File: semisupervised.py
Project: autograder_test_files
File Created: September 2020
Author: Shalini Chaudhuri (you@you.you)
Updated: September 2022, Arjun Agarwal
'''
import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

def complete_(data): # [1pts]
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels 
    Return:
        labeled_complete: n x (D+1) array (n <= N) where values contain both complete features and labels
    """
    # print(data)
    featureTruths = np.all(~np.isnan(data[:, :-1]), axis=1)
    labelTruths = ~np.isnan(data[:, -1])
    rV = data[np.logical_and(featureTruths, labelTruths)]
    # print(rV)
    return rV   
 
def incomplete_(data): # [1pts]
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_incomplete: n x (D+1) array (n <= N) where values contain incomplete features but complete labels
    """   
    # print(data)
    featureTruths = ~np.all(~np.isnan(data[:, :-1]), axis=1)
    labelTruths = ~np.isnan(data[:, -1])
    rV = data[np.logical_and(featureTruths, labelTruths)]
    # print(rV)
    return rV    

def unlabeled_(data): # [1pts]
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels   
    Return:
        unlabeled_complete: n x (D+1) array (n <= N) where values contain complete features but incomplete labels
    """
    featureTruths = np.all(~np.isnan(data[:, :-1]), axis=1)
    labelTruths = np.isnan(data[:, -1])
    rV = data[np.logical_and(featureTruths, labelTruths)]
    return rV

class CleanData(object):
    def __init__(self): # No need to implement
        pass

    def pairwise_dist(self, x, y): # [0pts] - copy from kmeans
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
        """
        return np.sqrt(np.maximum(np.sum(x ** 2, axis = 1, keepdims = True) + np.sum(y ** 2, axis = 1) - 2 * np.dot(x, y.T), 0))
    
    def __call__(self, incomplete_points,  complete_points, K, **kwargs): # [7pts]
        """
        Function to clean or "fill in" NaN values in incomplete data points based on
        the average value for that feature for the K-nearest neighbors in the complete data points. 

        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points:   N_complete   x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_complete + N_incomplete) x (D+1) numpy array, containing both the complete points and recently filled points

        Notes: 
            (1) The first D columns are features, and the last column is the class label
            (2) There may be more than just 2 class labels in the data (e.g. labels could be 0,1,2 or 0,1,2,...,M)
            (3) There will be at most 1 missing feature value in each incomplete data point (e.g. no points will have more than one NaN value)
            (4) You want to find the k-nearest neighbors, from the complete dataset, with the same class labels;
            (5) There may be missing values in any of the features. It might be more convenient to address each feature at a time.
            (6) Do NOT use a for-loop over N_incomplete; you MAY use a for-loop over the M labels and the D features (e.g. omit one feature at a time) 
            (7) You do not need to order the rows of the return array clean_points in any specific manner
        """
        # print(complete_points)
        # print(incomplete_points)
        # print(K)
        labels = np.unique(incomplete_points[:, -1])
        rV = []
        for label in labels:
            # print("Y")
            incompleteRows = incomplete_points[np.where(incomplete_points[:, -1] == label)]
            completeRows = complete_points[np.where(complete_points[:, -1] == label)]
            for features in range(len(complete_points[0]) - 1):
                #then we fix each one at a time! yay. 
                featureCol = completeRows[:, features]
                kVals = np.partition(featureCol, -K)[-K:]
                repVal = np.mean(kVals)
                replaceCols = np.isnan(incompleteRows[:, features])
                incompleteRows[replaceCols, features] = repVal
            if len(rV) == 0:
                rV = np.vstack((incompleteRows, completeRows)) 
            else:
                rV = np.vstack((incompleteRows, completeRows, rV)) 
        # print("J")
        # print(rV)
        return rV

        # raise NotImplementedError

def mean_clean_data(data): # [2pts]
    """
    Args:
        data: N x (D+1) numpy array where only last column is guaranteed non-NaN values and is the labels
    Return:
        mean_clean: N x (D+1) numpy array where each NaN value in data has been replaced by the mean feature value
    Notes: 
        (1) When taking the mean of any feature, do not count the NaN value
        (2) Return all values to max one decimal point
        (3) The labels column will never have NaN values
    """
    for i in range(len(data[0]) - 1):
        vals = data[:, i][~np.isnan(data[:, i])]
        data[np.isnan(data[:, i]), i] = np.mean(vals)
    return np.round(data, decimals=1)

class SemiSupervised(object):
    def __init__(self): # No need to implement
        pass
    
    def softmax(self,logit): # [0 pts] - can use same as for GMM
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array where softmax has been applied row-wise to input logit
        """
        maxVal = np.max(logit, axis = 1).reshape(-1,1)
        raise2E = np.exp(logit - maxVal)
        rowSum = np.sum(raise2E, keepdims=True, axis = 1).reshape(-1,1)
        return raise2E/rowSum
        # raise NotImplementedError

    def logsumexp(self,logit): # [0 pts] - can use same as for GMM
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:])
        """
        maxVal = np.max(logit, axis = 1).reshape(-1,1)
        raisedE = np.exp(logit-maxVal)
        sumExp = np.sum(raisedE, keepdims=True, axis = 1)
        rowSum = np.log(sumExp).reshape(-1,1)+maxVal
        return rowSum
    
    def normalPDF(self, logit, mu_i, sigma_i): # [0 pts] - can use same as for GMM
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.diagonal() should be handy.
        """
        D = logit.shape[1]
        sigma_i = np.diag(np.diag(sigma_i))
        determinant = np.linalg.det(sigma_i)
        
        return (1/np.power(2*np.pi, D/2)) * np.power(determinant, -0.5) * np.exp(np.sum(np.dot((logit - mu_i) * -0.5, np.linalg.inv(sigma_i)).T * (logit - mu_i).T, axis=0))
        return p1 * p2 * p3

    
    def _init_components(self, points, K, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: Nx(D+1) numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K; contains the prior probabilities of each class k
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Hint:
            As explained in the algorithm, you need to calculate the values of mu, sigma and pi based on the labelled dataset
        """
        pi = np.zeros(K)
        D = points.shape[1] - 1
        sigma = np.full((K, D, D), 0, float)
        mu = np.full((K, D), 0, float)
        copyPts = points
        for i in range (K):
            pts = (copyPts[:, -1] == i)
            pi[i] = np.sum(pts)/len(copyPts)

            pts2 = (copyPts[pts, :][:, :-1])
            mu[i] = np.mean(pts2, axis = 0)
            sigma[i]= np.diag(np.diag(np.cov(pts2, rowvar = False, bias = True)))
        return pi, mu, sigma
        raise NotImplementedError

    def _ll_joint(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
        """
        K = self.K
        N = len(self.points)
        # if full_matrix is True:
        llOutput = np.zeros((N, K))

        for i in range( K):
            llOutput[:, i] = np.log(pi[i] + LOG_CONST)
            llOutput[:, i] += np.log(self.multinormalPDF(self.points, mu[i], sigma[i]) + LOG_CONST)
            # outputLog = np.log(llOutput + LOG_CONST)
            # transposeOutput = np.transpose(outputLog)
            # piLog = np.log(pi + LOG_CONST)

        # return piLog + transposeOutput
        return llOutput
        raise NotImplementedError

    def _E_step(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        # if full_matrix is True:
        llOutput = self._ll_joint(pi, mu, sigma, full_matrix)
        gamma = self.softmax(llOutput)
        return gamma
        raise NotImplementedError

    def _M_step(self, points, gamma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. 
            
        Hint:  There are formulas in the slide.
        """
        gammaSum = np.sum(gamma, axis = 0)
        K = self.K
        D = len(self.points[0])

        # if full_matrix is True:
        mu = np.zeros((K, D))
        sigma =  np.zeros((K, D, D))

        for i in range (K):
            gammaVal = gamma[:,i].reshape(len(self.points),1)
            mu[i] = np.sum(gammaVal * self.points, axis = 0)/gammaSum[i]

            diffM = gammaVal*(self.points - mu[i])
            diffMtranspose = np.transpose(self.points - mu[i])
            dotDiffs = np.dot(diffMtranspose, diffM)

            sigma[i] = (dotDiffs)/gammaSum[i]
            # print(sigma)
        combinedGamma = np.sum(gamma, axis = 0)
        pi = (combinedGamma/len(self.points))

        return pi, mu, sigma
        raise NotImplementedError

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: N x (D+1) numpy array, where 
                - N is # points, 
                - D is the number of features,
                - the last column is the point labels (when available) or NaN for unlabeled points
            K: integer, number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            pi, mu, sigma: (1xK np array, KxD numpy array, KxDxD numpy array)
        """

        raise NotImplementedError


class ComparePerformance(object):

    def __init__(self): #No need to implement
        pass

    @staticmethod
    def accuracy_semi_supervised(training_data, validation_data, K:int) -> float: # [2.5 pts]
        """
        Train a classification model using your SemiSupervised object on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data 

        Args:
            training_data: N_t x (D+1) numpy array, where 
                - N_t is the number of data points in the training set, 
                - D is the number of features, and 
                - the last column represents the labels (when available) or a flag that allows you to separate the unlabeled data.
            validation_data: N_v x(D+1) numpy array, where 
                - N_v is the number of data points in the validation set,
                - D is the number of features, and 
                - the last column are the labels
            K: integer, number of clusters for SemiSupervised object
        Return:
            accuracy: floating number
        
        Note: (1) validation_data will NOT include any unlabeled points
              (2) you may use sklearn accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
        """

        raise NotImplementedError

    @staticmethod
    def accuracy_GNB(training_data, validation_data) -> float: # [2.5 pts]
        """
        Train a Gaussion Naive Bayes classification model (sklearn implementation) on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data 

        Args:
            training_data: N_t x (D+1) numpy array, where 
                - N is the number of data points in the training set, 
                - D is the number of features, and 
                - the last column represents the labels
            validation_data: N_v x (D+1) numpy array, where 
                - N_v is the number of data points in the validation set,
                - D is the number of features, and 
                - the last column are the labels
        Return:
            accuracy: floating number

        Note: (1) both training_data and validation_data will NOT include any unlabeled points
              (2) use sklearn implementation of Gaussion Naive Bayes: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        """

        training = GaussianNB().fit(training_data[:,:-1], training_data[:,-1])
        return training.score(validation_data[:,:-1],validation_data[:,-1])
    
        # raise NotImplementedError


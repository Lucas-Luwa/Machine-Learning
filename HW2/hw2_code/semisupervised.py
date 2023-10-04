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
        print(complete_points)
        print(incomplete_points)
        print(K)

        return 0
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
        raise NotImplementedError

    def logsumexp(self,logit): # [0 pts] - can use same as for GMM
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:])
        """
        raise NotImplementedError
    
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
        
        raise NotImplementedError
    
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

        raise NotImplementedError


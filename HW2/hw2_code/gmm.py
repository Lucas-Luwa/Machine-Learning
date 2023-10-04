import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        maxVal = np.max(logit, axis = 1).reshape(-1,1)
        raise2E = np.exp(logit - maxVal)
        rowSum = np.sum(raise2E, keepdims=True, axis = 1).reshape(-1,1)
        return raise2E/rowSum

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        maxVal = np.max(logit, axis = 1).reshape(-1,1)
        raisedE = np.exp(logit-maxVal)
        sumExp = np.sum(raisedE, keepdims=True, axis = 1)
        # print(logit)
        rowSum = np.log(sumExp).reshape(-1,1)+maxVal
        # print(rowSum)
        # print("X",maxVal)
        return rowSum

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        #No need to implement
        raise NotImplementedError

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
        Background:
            In the above calculation, you must avoid computing a $(N,N)$ matrix. Using the above equation for large N will crash your kernel 
            and/or give you a memory error on Gradescope. Instead, you can do this same operation by calculating $(X-\mu)\Sigma^{-1}$, a $(N,D)$ 
            matrix, transpose it to be a $(D,N)$ matrix and do an element-wise multiplication with $(X-\mu)^T$, which is also a $(D,N)$ matrix. 
            Lastly, you will need to sum over the 0 axis to get a $(1,N)$ matrix before proceeding with the rest of the calculation. 
            This uses the fact that doing an element-wise multiplication and summing over the 0 axis is the same as taking the diagonal of the $(N,N)$ matrix 
            from the matrix multiplication.
        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """
        try: sigInv = np.linalg.inv(sigma_i); 
        except: sigInv = np.linalg.inv(sigma_i + SIGMA_CONST);
        D = len(points[0])
        frontConst = 1 / ((2*np.pi) ** ((D)/2))

        sigmaDeterminant =np.linalg.det(sigma_i)
        sigmaConst = 1 / (np.sqrt(sigmaDeterminant))

        diffTranspose = np.transpose(points - mu_i)
        diffDotSigma = np.dot((points - mu_i), sigInv)
        transposeDot = np.transpose(diffDotSigma)
        sumDotProd = np.sum(transposeDot*diffTranspose, axis = 0)
        raiseExp = np.exp(-(1/2)*sumDotProd)

        return frontConst*sigmaConst*raiseExp

    def create_pi(self):
        """
        Initialize the prior probabilities 
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        K = self.K
        return np.full((1, K), 1/K)[0]

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        K = self.K
        D = len(self.points)
        selectedpts = np.random.choice(D, K, False)
        return self.points[selectedpts]
    
    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the 
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """ 
        K = self.K
        D = len(self.points[1])
        sigma = np.zeros((K, D, D))

        for i in range (0, K):
            sigma[i] = np.eye(D, D)

        return np.array(sigma)
    
    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5) #Do Not Remove Seed
        return self.create_pi(), self.create_mu(), self.create_sigma()

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        # === graduate implementation
        K = self.K
        D = len(self.points)
        if full_matrix is True:
            llOutput = np.zeros((K, D))

            for i in range(0, K):
                llOutput[i] = self.multinormalPDF(self.points, mu[i], sigma[i])
            outputLog = np.log(llOutput + LOG_CONST)
            transposeOutput = np.transpose(outputLog)
            piLog = np.log(pi + LOG_CONST)

        return piLog + transposeOutput

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        if full_matrix is True:
            llOutput = self._ll_joint(pi, mu, sigma, full_matrix)
            gamma = self.softmax(llOutput)
        return gamma

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        gammaSum = np.sum(gamma, axis = 0)
        K = self.K
        D = len(self.points[0])

        if full_matrix is True:
            mu = np.zeros((K, D))
            sigma =  np.zeros((K, D, D))

            for i in range (K):
                gammaVal = gamma[:,i].reshape(-1,1)
                mu[i] = np.sum(gammaVal * self.points, axis = 0)/gammaSum[i]

                diffM = (self.points - mu[i])*gammaVal
                diffMtranspose = np.transpose(self.points - mu[i])
                dotDiffs = np.dot(diffMtranspose, diffM)

                sigma[i] = (dotDiffs)/gammaSum[i]
                # print(sigma)
            combinedGamma = np.sum(gamma, axis = 0)
            pi = (combinedGamma/len(self.points))

        return pi, mu, sigma

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)


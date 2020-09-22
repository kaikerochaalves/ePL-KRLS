"""
    Author: Rafael Giordano Vieira
    Updated at: October 20, 2018
    Python version: 3.6
"""

import numpy as np
import gprof2dot


class ePLKRLS(object):

    """
    Evolving Participatory Learning with Kernel Recursive Least-Squares (ePL-KRLS).
    1. Create a new instance and provide the model parameters
    2. Call the evolve() method to make predictions.
    """   

    class KRLS(object):

        """
        Kernel Recursive Least-Squares Regression.
        1. Create a new instance, provide a kernel and the model parameters
        2. Call the update() method with one or more samples a time.
        3. Call the query() method to get estimations.
        """   

        class GaussianKernel(object):

            """
            Gaussian Kernel
            1. Call the kernel() method to computes the Gaussian kernel.
            """   
            
            def __init__(self, sigma=1.4):
                # Setting initial kernel size
                self.sigma = sigma

            def __call__(self, X, Z):
                return self.kernel(X, Z, self.sigma)

            @classmethod
            def kernel(cls, X, Z, sigma):
                # Computes the Gaussian kernel for the matrices X and Z.
                X = np.matrix(X, dtype="float32")
                Z = np.matrix(Z, dtype="float32")
                # Normalize the matrices between different bandwidths
                if hasattr(sigma, '__iter__'):
                   X /= 1.4142 * (np.array(sigma))
                   Z /= 1.4142 * (np.array(sigma))
                   sigma = 1.0
                else:
                   sigma = float(sigma)
                n, m = X.shape[0], Z.shape[0]
                XX = np.multiply(X, X)
                XX = XX.sum(axis = 1)
                ZZ = np.multiply(Z, Z)
                ZZ = ZZ.sum(axis = 1)
                d = np.tile(XX, (1, m)) + np.tile(ZZ.T, (n, 1)) - 2 * X * Z.T
                Kexpd = np.exp(-d.T / (2 * sigma * sigma))
                return np.array(Kexpd)

        # Initialization of a new instance of KRLS.
        def __init__(self, params={}):
            # Instance of a kernel to use
            self.kernel = self.GaussianKernel()        
            # Approximate linear dependence threshold [0,1]
            self.adopt_thresh = 0.01  
            # Other supporting variables
            self.alpha = None
            self.p = [[1]]
            # Intrinsic regularization (ridge term coefficient)
            self.gamma = 0.001
            # The dictionary of relevant states
            self.dico = None
            # The associated targets
            self.target = None
            # The kernel matrix
            self.k = None
            # The inverse kernel matrix
            self.kinv = None
            # How fast we want to forget what we did see
            self.forget_rate = 0.0
            # Index when we added what
            self.dico_idx = []
            # Allow overriding the attributes through the params dictionary
            for k in params:
                if hasattr(self, k):
                    setattr(self, k, params[k])
        
        def setup(self, sample, target):
            # Initializes everything with a first sample and target.
            if self.dico != None:
                return
            self.dico = sample
            self.target = target
            self.dico_idx.append(len(self.dico_idx))
            self.p = [[1]]
            self.k = self.gamma + self.kernel(sample, sample).T
            self.kinv = 1 / self.k
            self.alpha = np.dot(self.kinv, target)
        
        def update(self, sample, target):
            # Update the model using a sample and a target.
            s = np.array(sample)
            t = np.array(target)
            # Checking if we have one or several samples.
            if len(s.shape) > 1:
                for i in range(len(s)):
                    self.update(s[i], t[i])
                return
            # Checking if we have already an element in our dictionary.
            if type(self.dico) == type(None):
                self.setup(sample, target)
                return
            # Evaluating if the sample is linearly dependent upon the dictionary.
            sample_eval = self.evaluate_sample(sample)
            if sample_eval['dt'] > self.adopt_thresh:
                # Adding the sample to the dictionary (momentanly).
                self.add_sample_to_dictionary(
                    sample, target, 
                    sample_eval['ktt'], sample_eval['ktwid'], 
                    sample_eval['at'], sample_eval['dt']
                )
                # Clean up the dictionary if the sample increases error
                if self.dico.shape[0] > 15:
                    # If we follow an adaptive cleanup strategy, remove the least
                    # relevant sample. If not, just remove the oldest sample.
                    idx = self.least_relevant_element_in_dictionary()
                    self.eliminate_element_in_dictionary(idx)
                    # Update the weights
                    self.alpha = np.dot(self.kinv, self.target)
                else:
                    # Update the kernel sizes
                    self.p = np.vstack(
                        [np.hstack([self.p, np.zeros((self.dico.shape[0] - 1, 1))]),
                        np.hstack([np.zeros((1, self.dico.shape[0] - 1)), [[1]]])]
                    )
                    self.alpha = np.dot(self.kinv, self.target)
            else:
                # Update the kernel sizes in order to not waste the sample.
                tmp = np.dot(self.p, sample_eval['at'])
                qt = tmp / ( 1 + np.dot(sample_eval['at'].T, tmp))
                self.p = self.p - np.dot(qt, tmp.T)
                self.alpha = self.alpha + np.dot(
                    self.kinv, 
                    qt * (target - np.dot(sample_eval['ktwid'].T, self.alpha))
                )

        def evaluate_sample(self, sample):
            # Evaluates a sample if it is ALD or not.
            if type(self.dico) == type(None):
                return {'ktt': None, 'ktwid': None, 'at': None, 'dt': 0.0}
            ktt = self.gamma + self.kernel(sample, sample).T
            ktwid = self.kernel(self.dico, sample).T
            at = np.dot(self.kinv, ktwid)
            dt = ktt - np.dot(ktwid.T, at)
            return {'ktt': ktt, 'ktwid': ktwid, 'at': at, 'dt': dt}
        
        def add_sample_to_dictionary(self, sample, target, 
            # Adds a sample to the dictionary
                ktt=None, ktwid=None, at=None, dt=None):
            self.dico = np.vstack([self.dico, sample])
            self.target = np.vstack([self.target, target])
            self.dico_idx.append(len(self.dico_idx))
            # Update the kernel and inverse kernel matrices
            if ktt == None:
                ktt = self.gamma + self.kernel(sample, sample).T
            if type(ktwid) == type(None):
                ktwid = self.kernel(self.dico, sample).T
            if type(at) == type(None):
                at = np.dot(self.kinv, ktwid)
            if type(dt) == type(None):
                dt = ktt - np.dot(ktwid.T, at)
            self.k = np.vstack([np.hstack([self.k, ktwid]), np.hstack([ktwid.T, ktt])])
            self.kinv = (1 / dt[0,0]) * np.vstack([ 
                np.hstack([dt[0,0] * self.kinv + np.dot(at, at.T), -at]), 
                np.hstack([-at.T, [[1]]])
            ])
        
        def least_relevant_element_in_dictionary(self):
            # Returns the least relevant element in dictionary
            if np.random.uniform(0,1) < self.forget_rate:
                return self.dico_idx.index(min(self.dico_idx))
            # Otherwise, compute the heuristics
            weights = abs(self.alpha.T[0] / np.diag(self.kinv[:-1,:-1]))
            return weights.argmin()
        
        def eliminate_element_in_dictionary(self, idx):
            # Removes the element at index idx [0 to nelems-1].
            self.k  = np.delete(np.delete(self.k, idx, 0), idx, 1)
            # Extract the row vector idx and eliminate the idx-th element from it
            f = np.delete(np.matrix(self.kinv[idx]), idx, 1)
            # Delete column and row idx from kinv, and update kinv using f
            kinv2 = np.delete(np.delete(self.kinv, idx, 0), idx, 1)
            self.kinv = np.array(kinv2 - (f.T*f) / self.kinv[idx,idx])
            # Eliminate the element idx in the dictionary and the target
            self.dico = np.delete(self.dico, idx, 0)
            self.target = np.delete(self.target, idx, 0)
            # Update the dictionary accordingly: eliminate the entry idx
            didx = self.dico_idx.pop(idx)
            ndico = np.array(self.dico_idx)
            self.dico_idx = list(np.where(ndico > didx, ndico - 1, ndico))
        
        def query(self, sample):
            # Capture estimations
            kernvals = self.kernel(sample, self.dico).T
            res = np.dot(kernvals, self.alpha)
            return res

    # Model initialization
    def __init__(self, **kwargs):
        # Setting rule base history
        self.rules = list()
        # Setting local models
        self.models = list()
        # Setting participatory learning clustering algorithm parameters
        self.params = {
            'alpha': kwargs.get('alpha', 0.01),
            'beta': kwargs.get('beta', 0.18),
            'gamma': kwargs.get('gamma', 0.82),
            'tau': kwargs.get('tau', 0.18),
            'r': kwargs.get('r', 0.04)
        }

    # Measures the compatibility between two samples x1 and x2
    def compat(self, x1, x2):
        return 1.00 - np.linalg.norm(x1 - x2)/len(x1)

    # Measures the arousal index 
    def arousal(self, a, p_max):
        return a + self.params['beta'] * (1 - p_max - a)

    # Measures the firing degree for a new sample in relation to a cluster 
    def gauss(self, x, v):
        return np.exp(-np.power(np.linalg.norm(v - x), 2)/(2*np.power(0.07, 2)))

    # Evolves the model (main method)
    def evolve(self, x, y=0.):
        # Creating extended input data
        x_ext = x
        # Checking for system prior knowledge
        if len(self.models) == 0:
            # Setting the sample as the first cluster
            self.models = [{'pos': x_ext, 'a': 0., 'coefs': self.KRLS()}]
            self.models[-1]['coefs'].update(x_ext, y)
            return y
        # Calculating the compatibility measure
        p = np.array([self.compat(x_ext, v['pos']) for v in self.models])
        # Degree of activation
        mu = np.array([self.gauss(x_ext, v['pos']) for v in self.models])
        # Calculating the output
        output = sum([np.array(
            i * j['coefs'].query(x_ext)[0]) for i, j in zip(
                mu, self.models)]) / sum(mu)
        # Calculating the arousal index
        for n, v in enumerate(self.models):
            v['a'] = self.arousal(v['a'], p[n])
        # Checking if arousal is greater than a threshold
        if min([i['a'] for i in self.models]) >= self.params['tau']:
            # Creating a new rule
            self.models = np.append(self.models, {
                'pos': x_ext, 'a': 0., 'coefs': self.KRLS()})
            self.models[-1]['coefs'].update(x_ext, y)
        else:
            # Looking for the most compatible rule
            s = self.models[np.argmax(p)]['pos']
            # Updating the rule
            self.models[np.argmax(p)]['pos'] = s + self.params['alpha'] * (
                np.power(max(p), 1.00 - self.models[np.argmax(p)]['a'])) * (x_ext - s)
            self.models[np.argmax(p)]['coefs'].update(x_ext, y)
        # Removing redundant clusters
        N = len(self.models)
        if N > 1:
            excl = []
            mapp = []
            for i in range(N - 1):
                for j in range(i + 1, N):
                    if i in excl or j in excl:
                        continue
                    p = self.compat(self.models[i]['pos'], self.models[j]['pos'])
                    if p >= self.params['gamma']:
                        excl.append(j)
                        mapp.append([i, j])
            for m in mapp:
                self.models[m[0]]['pos'] = (
                    self.models[m[0]]['pos'] + self.models[m[1]]['pos'])/2  
                self.models[m[0]]['a'] = (
                  self.models[m[0]]['a'] + self.models[m[1]]['a'])/2 

            self.models = np.delete(self.models, excl)
        
        # Calculating statistics for a step k
        self.rules.append(len(self.models))
        if output[0] > 1:
            return 1./np.absolute(output[0])
        elif output[0] < 0:
            return np.absolute(output[0])
        return output[0]
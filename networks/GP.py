import tensorflow as tf
import numpy as np
from .utils import * 

class GP(object):
    def __init__(self, D_IN, D_OUT, M=50):
        """
        Initialize GP layer 
        D_IN    : dimension of input
        D_OUT   : dimension of output
        M       : the number of inducing points
        """
        self.sig_offset = 1e-5
        self.process_sig_offset = 1e-5
        self.lambda_offset = 1e-5
        self.D_IN = D_IN
        self.D_OUT = D_OUT
        self.M = M
        
        # Define mean function
        self._init_mean_function(D_IN, D_OUT)
        
        # Kernel Parameters (SE ARE kernel)
        self.ARD_loglambda = tf.Variable(np.zeros([1,D_IN]), dtype=tf.float32) # (1,Din)
        self.ARD_lambda = tf.exp(self.ARD_loglambda) + self.lambda_offset
        self.ARD_logsig0 = tf.Variable(0.0, dtype=tf.float32)
        self.ARD_var0 = (tf.exp(self.ARD_logsig0) + self.sig_offset)**2
            
        # Inducing Points (Z -> U)
        self.Z = tf.Variable(np.random.uniform(-3,3,[M,D_IN]), dtype=tf.float32) # (M,Din)
        self.GPmean = self.mean(self.Z)
        self.U_mean = tf.Variable(np.random.uniform(-3,3,[M,D_OUT]), dtype=tf.float32) # np.random.uniform(-2,2,[M,D_OUT])  # (M,Dout)
        self.U_logL_diag = tf.Variable(np.zeros([D_OUT,M]), dtype=tf.float32)  # (Dout,M)
        self.U_L_diag = tf.exp(self.U_logL_diag)
        self.U_L_nondiag = tf.Variable(np.zeros([D_OUT,int(M*(M-1)/2)]), dtype=tf.float32)  # (Dout,M(M-1)/2)    
        self.U_L = tf.matrix_set_diag(vecs_to_tri(self.U_L_nondiag, M), self.U_L_diag) # (Dout,M,M)
        self.U_cov = (self.sig_offset**2)*tf.eye(M,batch_shape=[D_OUT]) + self.U_L@tf.transpose(self.U_L,perm=(0,2,1)) # (Dout,M,M)
        
        # Covariance among inducing points
        self.Kzz = SEARD(self.Z,self.Z,self.ARD_lambda,self.ARD_var0,M,M,D_IN) + (self.sig_offset**2)*tf.eye(M) # (M,M)
        self.Kzz_L = tf.cholesky(self.Kzz)
        self.Kzz_L_inv = tf.matrix_triangular_solve(self.Kzz_L,tf.eye(M),lower=True) # tf.matrix_inverse(self.Kzz_L)
        self.Kzz_inv = tf.transpose(self.Kzz_L_inv)@self.Kzz_L_inv
        
            
        # Processing Noise
        self.logbeta = tf.Variable(np.zeros([1,D_OUT]), dtype=tf.float32) # set as 0.0
        self.beta = tf.exp(self.logbeta) + self.process_sig_offset # (1,Dout)
        self.beta_expand = tf.expand_dims(self.beta, axis=0) # (1,1,Dout)
    def _init_mean_function(self,D_IN, D_OUT):
        # self.H = W = tf.Variable(np.zeros([D_IN,D_OUT]), dtype=tf.float32)
        # self.B = tf.Variable(np.zeros([1,D_OUT]), dtype=tf.float32) # (1,Dout)

        # H0 = np.zeros([D_IN,D_OUT])     
        # self.H = W = tf.constant(H0, dtype=tf.float32)
        # self.B = tf.constant(np.zeros([1,D_OUT]), dtype=tf.float32) # (1,Dout)
      
        initializer = tf.contrib.layers.xavier_initializer()
        self.H = W = tf.Variable(initializer([D_IN,D_OUT]))
        self.B = tf.Variable(np.zeros([1,D_OUT]), dtype=tf.float32) # (1,Dout)
    
    def mean(self, X):
        # X : (N,D_in)
        return X@self.H + self.B 
    
    def predict(self, x, noise=True):
        # X : (N,D_IN)
        N = tf.shape(x)[0]
        Kzx = SEARD(self.Z, x, self.ARD_lambda, self.ARD_var0, self.M, N, self.D_IN)
        
        f_mu, f_var = GPpredict(self.mean(x), self.GPmean, self.U_mean, self.U_cov
                                ,self.Kzz, self.Kzz_inv, Kzx, self.ARD_var0
                                ,N, self.M, self.D_OUT)
        f_var = tf.clip_by_value(f_var, 1e-5, 1e5)
        
        epsilon = tf.random_normal((N,self.D_OUT)) # (N,D_OUT)
        f = f_mu + tf.sqrt(f_var)*epsilon # (N,D_OUT)
        if noise:
            f = f + tf.random_normal((N,self.D_OUT))*self.beta # (N,D_OUT)
        return f
    
     
    def KL(self):
        """
        Compute KL(q(U),p(U)), where q(U)~MN(mu0,cov0,Sig), p(U)~MN(mu1,cov1,Sig)
        mu0, mu1   : (M,D)
        cov0, cov1 : (M,M)
        Sig        : (D,D)
        """
        
        Kzz_invtile = expand_tile(self.Kzz_inv,self.D_OUT)
        term1 = tf.reduce_sum(Kzz_invtile*self.U_cov,axis=[1,2]) # (D,)
        mu10 = tf.expand_dims(tf.transpose(self.GPmean - self.U_mean),axis=2) # (D,M,1)
        mu10T = tf.transpose(mu10, perm=[0,2,1]) # (D,1,M)
        term2 = tf.squeeze(mu10T@Kzz_invtile@mu10,axis=[1,2]) # (D,)
        term3_1 = -logdet(self.U_L) # (D,)
        term3_2 = -logdet(expand_tile(self.Kzz_L_inv,self.D_OUT)) # (D,)
        return 0.5 * tf.reduce_sum(term1 + term2 + term3_1 + term3_2 - self.M)

    
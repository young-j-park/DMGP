import numpy as np
import tensorflow as tf
from .utils import * 
from .MGP import * 
from .GP import * 

class DGP_DVSI():
    def __init__(self, N_LAYER, D, M, K=100, Mat=True, MCO=True):
        """
        Construct DGP_DVSI network.
        N_LAYER : the number of gp layers
        D       : dimension of input/output of each layer
        M       : the number of inducing points of each layer
        """
        self.N_LAYER = N_LAYER
        self.D = D # (N_LAYER+1,)
        self.M = M # (N_LAYER,)
        self.K = K
        self.MCO = MCO
        
        # Define placeholder for output, Y
        self.X = tf.placeholder(tf.float32, shape=(None,D[0])) # (N,D[0])
        self.Y = tf.placeholder(tf.float32, shape=(None,D[-1])) # (N,D[-1])
        
        N = tf.shape(self.X)[0]
        self.N = N
            
        # Initialize gp layers
        self.gp_layers = []
        for i in range(self.N_LAYER):
            if Mat:
                self.gp_layers.append(MGP(D[i],D[i+1],M[i])) 
            else:
                self.gp_layers.append(GP(D[i],D[i+1],M[i]))
                
        # Sample K samples of F|X
        
        Xsample_all = tf.reshape(expand_tile(self.X, K), (-1,self.D[0]))
        for i in range(self.N_LAYER-1):
            Xsample_all = self.gp_layers[i].predict(Xsample_all)
        Fsample_temp = self.gp_layers[-1].predict(Xsample_all, noise=False)
        self.Fsample = tf.reshape(Fsample_temp,(K,N,D[-1]))
        
        # Corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables and saver
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        
        ### Launch the session
        self.sess.run(init)
        
    def _create_loss_optimizer(self):
        # Create Loss function
        logweights_temp = 0.5*(self.Fsample - tf.expand_dims(self.Y,axis=0))**2 / self.gp_layers[-1].beta**2 \
                    + tf.log(self.gp_layers[-1].beta) + 0.5*tf.log(2*np.pi)
        self.logweights = tf.reduce_sum(logweights_temp,axis=2) # (K,N)
        logweights_min = tf.reduce_min(self.logweights, axis=0, keepdims=True) # (1,N)
        weights = tf.exp(-self.logweights+logweights_min) # (K,N)
        weights_norm = tf.reduce_sum(weights,axis=0,keepdims=True) # (1,N)
        self.weights_constant = tf.stop_gradient(weights/weights_norm) # (K,N)
        
        self.kl = 0.0
        for i in range(self.N_LAYER):
            kl_temp = self.gp_layers[i].KL()
            self.kl = self.kl + kl_temp
            
        if self.MCO:
            # self.loss = tf.reduce_sum(self.weights_constant*self.logweights) + self.kl
            self.elbo = tf.reduce_sum(tf.log(weights_norm/self.K) - logweights_min) - self.kl
            self.loss = -self.elbo
        else:
            self.loss = tf.reduce_sum(self.logweights)/self.K + self.kl
            self.elbo = -self.loss
            
        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        # grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in grads_and_vars]
        # self.opt = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.opt = self.optimizer.minimize(self.loss)
        
    def train(self,Xtrain,Ytrain,Xtest,Ytest,epoch=500):
        elbo_hist = []
        elbo_hist_test = []
        for i in range(epoch):
            _, loss, elbo = self.sess.run([self.opt, self.loss, self.elbo], feed_dict={self.X:Xtrain, self.Y:Ytrain})
            elbo_test = self.sess.run(self.elbo, feed_dict={self.X:Xtest, self.Y:Ytest})
            elbo_hist.append(elbo)
            elbo_hist_test.append(elbo_test)
            
            if i % 100 == 0:
                print(i,loss, elbo)
        return elbo_hist, elbo_hist_test
    
    def predict(self,Xtest_numpy, K=1000):
        Xtest = tf.convert_to_tensor(Xtest_numpy, dtype=tf.float32)
        
        N = tf.shape(Xtest)[0]
        D = tf.shape(Xtest)[1]
        
        Xsample_all = tf.tile(Xtest, (K,1)) # (KN,Din)
        for i in range(self.N_LAYER):
            Xsample_all = self.gp_layers[i].predict(Xsample_all)

        Ysample_all = tf.reshape(Xsample_all,(K,N,-1)) # (K,N,Dout)
        Ysample_mean, Ysample_var = tf.nn.moments(Ysample_all, axes=0) # (N,Dout)
        
        return self.sess.run([Ysample_mean, Ysample_var])
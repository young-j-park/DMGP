import numpy as np
import tensorflow as tf

def compute_MKL(mu0, cov0, L0, mu1, cov1_inv, L1_inv, Sig_inv, M, D):
    """
    Compute -KL(P,Q), where P~MN(mu0,cov0,Sig), Q~MN(mu1,cov1,Sig)
    mu0, mu1   : (M,D)
    cov0, cov1 : (M,M)
    Sig        : (D,D)
    """
    term1 = D*tf.reduce_sum(cov1_inv*cov0)
    mu10 = mu1 - mu0 # (M,D)
    term2_1 = mu10@Sig_inv # (M,D)
    term2_2 = cov1_inv@mu10 # (M,D)
    term2 = tf.reduce_sum(term2_1*term2_2)
    term3_1 = -logdet(L0)
    term3_2 = -logdet(L1_inv)
    return 0.5*(term1 + term2 + term3_1*D + term3_2*D - M*D)

def compute_KL(mu0, cov0, L0, mu1, cov1_inv, L1_inv, M):
    """
    Compute KL(P,Q), where P~N(mu0,cov0), Q~N(mu1,cov1)
    mu0, mu1   : (M,D)
    cov0, cov1 : (D,M,M)
    """
    term1 = tf.reduce_sum(cov1_inv*cov0,axis=[1,2]) # (D,)
    mu10 = tf.expand_dims(tf.transpose(mu1 - mu0),axis=2) # (D,M,1)
    mu10T = tf.transpose(mu10, perm=[0,2,1]) # (D,1,M)
    term2 = tf.squeeze(mu10T@cov1_inv@(mu10),axis=[1,2]) # (D,)
    term3_1 = -logdet(L0) # (D,)
    term3_2 = -logdet(L1_inv) # (D,)
    return 0.5 * tf.reduce_sum(term1 + term2 + term3_1 + term3_2 - M)

def logdet(L):
    """
    Compute logdet(X), where X = L@L^T (L: cholesky)
    L : (...,M,M)
    """
    Ldiag = tf.matrix_diag_part(L) # (...,M) 
    logLdiag = tf.log(tf.abs(Ldiag)) # (...,M)
    ldet = tf.reduce_sum(logLdiag,axis=-1) # (...,)
    return 2*ldet

def expand_tile(X,multiply):
    """
    Expand the first dimension and tile.
    X must be 2-dim tensor: e.g.(B,D).
    Return tensor has a shape of (multiply,B,D)
    """
    return tf.tile(tf.expand_dims(X,axis=0),(multiply,1,1))

def vecs_to_tri(vectors, N):
    """
    Takes a D x M tensor `vectors' and maps it to a D x matrix_size X matrix_sizetensor
    where the where the lower triangle of each matrix_size x matrix_size matrix is
    constructed by unpacking each M-vector.
    Native TensorFlow version of Custom Op by Mark van der Wilk.
    def int_shape(x):
        return list(map(int, x.get_shape()))
    D, M = int_shape(vectors)
    N = int( np.floor( 0.5 * np.sqrt( M * 8. + 1. ) - 0.5 ) )
    # Check M is a valid triangle number
    assert((matrix * (N + 1)) == (2 * M))
    """
    indices = list(zip(*np.tril_indices(N,-1)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

    def vec_to_tri_vector(vector):
        return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

    return tf.map_fn(vec_to_tri_vector, vectors)

def vec_to_tri(vector, N):
    """
    Takes a D x M tensor `vectors' and maps it to a D x matrix_size X matrix_sizetensor
    where the where the lower triangle of each matrix_size x matrix_size matrix is
    constructed by unpacking each M-vector.
    Native TensorFlow version of Custom Op by Mark van der Wilk.
    def int_shape(x):
        return list(map(int, x.get_shape()))
    D, M = int_shape(vectors)
    N = int( np.floor( 0.5 * np.sqrt( M * 8. + 1. ) - 0.5 ) )
    # Check M is a valid triangle number
    assert((matrix * (N + 1)) == (2 * M))
    """
    indices = list(zip(*np.tril_indices(N,-1)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)
    return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

def SEARD(x1,x2,ARD_lambda,ARD_var0,M1,M2,D):
    """
    SE ARD(Squared Exponential Automatic Relevance Determination) Kernel
    k(x1,x2) = ARD_var0*exp(-0.5*\Sum_{d=1}^{D}(ARD_lambda[d]*(x1-x2)^2))
    x1 : (M1,D)
    x2 : (M2,D)
    """
    X1 = tf.reshape(tf.tile(x1,(1,M2)),shape=(M1*M2,D)) # (M1*M2,D)
    X2 = tf.tile(x2, (M1,1)) # (M1*M2,D)
    X1X2 = X1-X2 # (M1*M2,D)

    r = tf.reduce_sum(ARD_lambda*X1X2**2, axis=-1)
    kxx_vec = ARD_var0*tf.exp(-0.5*r)
    return tf.reshape(kxx_vec,shape=(M1,M2)) # (M1,M2) tensor

def MGPpredict(mX,mZ,m,S,Kzz,Kzz_inv,Kzx,Kxx,N,M,D_OUT):
    """
    Compute q(f|m,S;X,Z)
    mX:= mean(X) : (N,D_OUT) / mZ:=mean(Z) : (M,D_OUT)
    m : (M,D_OUT) / S : (M,M)
    Kzz : (M,M) /  Kzz_inv : (M,M) / Kzx : (M,N) / Kxx : scalar
    """
    alpha_temp = tf.transpose(Kzz_inv@Kzx) # (N,M)
    f_mu = mX + alpha_temp@(m-mZ) # (N,D_OUT)
    
    alpha = tf.expand_dims(alpha_temp,axis=2) # (N,M,1)
    alphaT = tf.transpose(alpha, perm=(0,2,1)) # (N,1,M)
    temp = expand_tile(Kzz-S,N) # (N,M,M)
    f_var = Kxx - tf.squeeze(alphaT@temp@alpha,axis=2) # (N,1)
    return f_mu, f_var

def GPpredict(mX,mZ,m,S,Kzz,Kzz_inv,Kzx,Kxx,N,M,D_OUT):
    """
    Compute q(f|m,S;X,Z). See eq.(9) of https://arxiv.org/abs/1705.08933
    mX:= mean(X) : (N,D_OUT) / mZ:=mean(Z) : (M,D_OUT)
    m : (M,D_OUT) / S : (D_OUT,M,M)
    Kzz : (M,M) /  Kzz_inv : (M,M) / Kzx : (M,N) / Kxx : scalar
    """
    alpha_temp = tf.transpose(Kzz_inv@Kzx) # (N,M)
    f_mu = mX + alpha_temp@(m-mZ) # (N,D_OUT)
    
    alpha_expand = tf.expand_dims(expand_tile(alpha_temp,D_OUT),axis=3) # (D_OUT,N,M,1)
    alpha = tf.transpose(alpha_expand, perm=(1,0,2,3)) # (N,D_OUT,M,1)
    alphaT = tf.transpose(alpha, perm=(0,1,3,2)) # (N,D_OUT,1,M)
    temp = expand_tile(Kzz,D_OUT)-S # (D_OUT,M,M)
    f_var = Kxx - tf.squeeze(alphaT@tf.tile(tf.expand_dims(temp,axis=0),(N,1,1,1))@alpha,axis=[2,3]) # (N,D_OUT)
    return f_mu, f_var

def mvnSample(m,v,L,N,D,K):
    """
    Sample K vectors from the distribution ~ N(m[i,:],v[i]*Sig) where Sig = LL^T
    m : (N,D) / v : (N,1), L : (D,D)
    """
    epsilon = tf.random_normal((K,N,D,1)) # (K,N,D,1)
    vexpand = tf.expand_dims(v,axis=0) # (1,N,1)
    Lexpand = tf.tile(tf.expand_dims(tf.expand_dims(L,axis=0),axis=0),(K,N,1,1)) # (K,N,D,D)
    return tf.expand_dims(m,axis=0) + tf.sqrt(vexpand)*tf.squeeze(Lexpand@epsilon,axis=3) # (K,N,D)

def Sample(m,v,N,D,K):
    """
    Sample K values from the distribution ~ N(m[i,d],v[i,d])
    m : (N,D) / v : (N,D)
    """
    epsilon = tf.random_normal((K,N,D)) # (K,N,D)
    return tf.expand_dims(m,axis=0) + tf.expand_dims(tf.sqrt(v),axis=0)*epsilon # (K,N,D)
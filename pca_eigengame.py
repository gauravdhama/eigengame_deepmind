# Importing necessary libraries
import numpy as np
import jax
import jax.numpy as jnp

# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
def calc_numpy_eig(X):
    p,q = jnp.linalg.eig(jnp.dot(jnp.transpose(X),X))
    return (p,q)

# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V1 holds the previously computed eigenvectors
def model(v,X,V1):
    M = jnp.dot(jnp.transpose(X),X)
    rewards = jnp.dot(jnp.transpose(v),jnp.dot(M,v))
    penalties = 0
    for j in range(np.size(V1[:,V1.any(0)],axis=1)-1):
        penalties = penalties + (jnp.dot(jnp.transpose(v),jnp.dot(M,V1[:,j].reshape(-1,1))))**2/jnp.dot(jnp.transpose(V1[:,j].reshape(-1,1)),jnp.dot(M,V1[:,j].reshape(-1,1)))
    return jnp.sum(rewards-penalties)

# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
def update(v,X,V1,lr=0.1,riemannian_projection=False):
    dv = jax.grad(model)(v,X,V1)
    if riemannian_projection:
        dvr = dv - (jnp.dot(dv.T,v))*v
        vhat = v+lr*dvr
    else:
        vhat = v+lr*dv
    return (vhat/jnp.linalg.norm(vhat))

# Run the update step iteratively across all eigenvectors
def calc_eigengame_eigenvectors(X,n,iterations=100):
    v = jnp.array([[1.0],[1.0],[1.0],[1.0]])
    v = v/jnp.linalg.norm(v)
    v0 = jnp.array([[1.0],[1.0],[1.0],[1.0]])
    v0 = v0/jnp.linalg.norm(v0)
    V1 = np.zeros_like(X)
    V1[:,0] = v.T

    for k in range(n):
        print ("Finding the eigenvector ",k)
        for i in range(iterations):
            if k==0:
                v = update(v,X,V1)
            else:
                #v = update(v,X,V1,riemannian_projection=True)
                v = update(v,X,V1)
        V1[:,k] = v.T
        v = v0
        if k<n-1:
            V1[:,k+1] = v0.T
    return V1

# Calculate eigenvalues once the eigenvectors have been computed
def calc_eigengame_eigenvalues(X,V1):
    M = jnp.dot(jnp.transpose(X),X)
    n = jnp.size(V1,axis=1)
    eigvals = np.zeros((1,n))
    for k in range(n):
        eigvals[:,k] = jnp.dot(V1[:,k],jnp.dot(M,V1[:,k].reshape(-1,1)))
    return eigvals

# Matrix X for which we want to find the PCA
X = jnp.array([[7.,4.,5.,2.],
            [2.,19.,6.,13.],
            [34.,23.,67.,23.],
            [1.,7.,8.,4.]])

# X = jnp.array([[9.,0.,0.,0.],
#             [0.,8.,0.,0.],
#             [0.,0.,7.,0.],
#             [0.,0.,0.,1.]])

# Centre the data
# X = X-jnp.mean(X,axis=0)
# print(X)

p,q = calc_numpy_eig(X)
V1 = calc_eigengame_eigenvectors(X,4)
print("\n Eigenvalues calculated using numpy are :\n",p)
print("\n Eigenvectors calculated using numpy are :\n",q)
print("\n Eigenvalues calculate using the Eigengame are :\n",calc_eigengame_eigenvalues(X,V1))
print("\n Eigenvectors calculated using the Eigengame are :\n",V1)
print("\n Squared error in estimation of eigenvectors as compared to numpy :\n",np.sum((np.abs(q)-np.abs(V1))**2,axis=0))

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sb  
from scipy.io import loadmat  
from scipy import stats  
from scipy.optimize import minimize


data = loadmat('ex8data1.mat')  
X = data['X']  
print (X.shape)  

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(X[:,0], X[:,1])  

#plt.show()

def estimate_gaussian(X):  
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)

    return mu, sigma

mu, sigma = estimate_gaussian(X)  

#print(mu, sigma ) 

Xval = data['Xval']  
yval = data['yval']

print(Xval.shape, yval.shape )

dist = stats.norm(mu[0], sigma[0])  
#print(dist.pdf(X[:,0])[0:50]  )

p = np.zeros((X.shape[0], X.shape[1]))  
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])  
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])

#print(p.shape)  

pval = np.zeros((Xval.shape[0], Xval.shape[1]))  
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])  
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])  

#print(pval)

def select_threshold(pval, yval):  
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon

        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)  
print(epsilon, f1 ) 


# indexes of the values considered to be outliers
outliers = np.where(p < epsilon)

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(X[:,0], X[:,1])  
ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')  

#plt.show()


data = loadmat('ex8_movies.mat')  
print(data)  

Y = data['Y']  
R = data['R']  
print(Y.shape, R.shape)

print(Y[1,R[1,:]].mean())

fig, ax = plt.subplots(figsize=(12,12))  
ax.imshow(Y)  
ax.set_xlabel('Users')  
ax.set_ylabel('Movies')  
fig.tight_layout()  

#plt.show()

def cost(params, Y, R, num_features, learning_rate):  
    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)

    # initializations
    J = 0
    X_grad = np.zeros(X.shape)  # (1682, 10)
    Theta_grad = np.zeros(Theta.shape)  # (943, 10)

    # compute the cost
    error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
    squared_error = np.power(error, 2)  # (1682, 943)
    J = (1. / 2) * np.sum(squared_error)

    # add the cost regularization
    J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
    J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))

    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad



users = 4  
movies = 5  
features = 3

params_data = loadmat('ex8_movieParams.mat')  
X = params_data['X']  
Theta = params_data['Theta']

X_sub = X[:movies, :features]  
Theta_sub = Theta[:users, :features]  
Y_sub = Y[:movies, :users]  
R_sub = R[:movies, :users]

params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

print(cost(params, Y_sub, R_sub, features, 1.5))

J, grad = cost(params, Y_sub, R_sub, features, 1.5)  
print(J, grad)


movie_idx = {}  
f = open('movie_ids.txt')  
for line in f:  
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

ratings = np.zeros((1682, 1))

ratings[0] = 4  
ratings[6] = 3  
ratings[11] = 5  
ratings[53] = 4  
ratings[63] = 5  
ratings[65] = 3  
ratings[68] = 5  
ratings[97] = 2  
ratings[182] = 4  
ratings[225] = 5  
ratings[354] = 5

print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))  



R = data['R']  
Y = data['Y']

Y = np.append(Y, ratings, axis=1)  
R = np.append(R, ratings != 0, axis=1)  


movies = Y.shape[0]  
users = Y.shape[1]  
features = 10  
learning_rate = 10.

X = np.random.random(size=(movies, features))  
Theta = np.random.random(size=(users, features))  
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

Ymean = np.zeros((movies, 1))  
Ynorm = np.zeros((movies, users))

for i in range(movies):  
    idx = np.where(R[i,:] == 1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]

fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, learning_rate),  
                method='CG', jac=True, options={'maxiter': 100})
print(fmin)  

X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))  
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

print(X.shape, Theta.shape)

predictions = X * Theta.T  
my_preds = predictions[:, -1] + Ymean  
sorted_preds = np.sort(my_preds, axis=0)[::-1]  
print(sorted_preds[:10])

idx = np.argsort(my_preds, axis=0)[::-1]  
print("Top 10 movie predictions:")  
for i in range(10):  
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))





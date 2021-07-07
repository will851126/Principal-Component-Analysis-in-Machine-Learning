# A Complete Guide to Principal Component Analysis — PCA in Machine Learning


## Principal Component Analysis
Principal Component Analysis or PCA is a widely used technique for dimensionality reduction of the large data set. Reducing the number of components or features costs some accuracy and on the other hand, it makes the large data set simpler, easy to explore and visualize. Also, it reduces the computational complexity of the model which makes machine learning algorithms run faster. It is always a question and debatable how much accuracy it is sacrificing to get less complex and reduced dimensions data set. we don’t have a fixed answer for this however we try to keep most of the variance while choosing the final set of components.

In this projrcts, we will be discussing the step by step approach to achieve dimensionality reduction using PCA 

## Steps Involved in PCA

1. Standardize the data. (with mean =0 and variance = 1)
2. Compute the Covariance matrix of dimensions.
3. Obtain the Eigenvectors and Eigenvalues from the covariance matrix (we can also use correlation matrix or even Single value decomposition, however in this post will focus on covariance matrix).
4. Sort eigenvalues in descending order and choose the top k Eigenvectors that correspond to the k largest eigenvalues (k will become the number of dimensions of the new feature subspace k≤d, d is the number of original dimensions).
5. Construct the projection matrix W from the selected k Eigenvectors.
6. Transform the original data set X via W to obtain the new k-dimensional feature subspace Y.


### 1. Standardization

When there are different scales used for the measurement of the values of the features, then it is advisable to do the standardization to bring all the feature spaces with mean = 0 and variance = 1.
The reason why standardization is very much needed before performing PCA is that PCA is very sensitive to variances. Meaning, if there are large differences between the scales (ranges) of the features, then those with larger scales will dominate over those with the small scales.
For example, a feature that ranges between 0 to 100 will dominate over a feature that ranges between 0 to 1 and it will lead to biased results. So, transforming the data to the same scales will prevent this problem. That is where we use standardization to bring the features with mean value 0 and variance 1.

In this article, I am using the Iris data set. Although all features in the Iris data set are measured in centimetres, Still I will continue with the transformation of the data onto the unit scale (mean=0 and variance=1), which is a requirement for the optimal performance of many machine learning algorithms. Also, it will help us to understand how this process works.


### 2. eigen decomposition — Computing Eigenvectors and Eigenvalues

The eigenvectors and eigenvalues of a covariance (or correlation) matrix represent the “core” of a PCA:
* The Eigenvectors (principal components) determine the directions of the new feature space, and the eigenvalues determine their magnitude.
* In other words, the eigenvalues explain the variance of the data along the new feature axes. It means the corresponding eigenvalue tells us that how much variance is included in that new transformed feature.
* To get eigenvalues and Eigenvectors we need to compute the covariance matrix. So in the next step let’s compute it.

### 2-1. Covariance Matrix
The classic approach to PCA is to perform the Eigen decomposition on the covariance matrix Σ, which is a d×d matrix where each element represents the covariance between two features. Note, d is the number of original dimensions of the data set. In Iris data set we have 4 features hence covariance matrix will be of order 4×4.

### 2-2 . Eigenvectors and Eigenvalues computation from the covariance matrix

Here if we know concepts of Linear Algebra and how to calculate Eigenvectors and Eigenvalues of the matrix then this is going to be very helpful in understanding the below concepts. So it would be advisable to go through some of the basic concepts of Linear Algebra to have a deeper understanding of how everything works.
Here I am using numpy array to calculate Eigenvectors and Eigenvalues of the standardized feature space values as following:

### 3. Selecting The Principal Components

* The typical goal of a PCA is to reduce the dimensionality of the original feature space by projecting it onto a smaller subspace, where the eigenvectors will form the axes.

* However, the eigenvectors only define the directions of the new axis, since they have all the same unit length 1.
So now the question comes that how to select the new set of Principal components. The rule behind is that we sort the Eigenvalues in descending order and then choose the top k features concerning top k Eigenvalues.
The idea here is that by choosing top k we have decided that the variance which corresponds to those k feature space is enough to describe the data set. And by losing the remaining variance of those not selected features, won’t cost the accuracy much or we are OK to loose that much accuracy that costs because of neglected variance.
So this is the decision which we have to make based on the problem set given and also based on the business case. There is no perfect rule to decide it.

### 3.1 Sorting eigenvalues
To decide which Eigenvector(s) can be dropped without losing too much information for the construction of lower-dimensional subspace, we need to inspect the corresponding eigenvalues:

* The Eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data; those are the ones can be dropped.
* To do so, the common approach is to rank the eigenvalues from highest to lowest to choose the top k Eigenvectors.

### 3.2 Explained Variance
* After sorting the eigenpairs, the next question is “how many principal components are we going to choose for our new feature subspace?”
* A useful measure is the so-called “explained variance,” which can be calculated from the eigenvalues.
* The explained variance tells us how much information (variance) can be attributed to each of the principal components.

### 4. Construct the projection matrix W from the selected k eigenvectors
* Projection matrix will be used to transform the Iris data onto the new feature subspace or we say newly transformed data set with reduced dimensions.

* It is a matrix of our concatenated top k Eigenvectors.

Here, we are reducing the 4-dimensional feature space to a 2-dimensional feature subspace, by choosing the “top 2” Eigenvectors with the highest Eigenvalues to construct our d×k-dimensional Eigenvector matrix W.

### 5. Projection Onto the New Feature Space

In this last step, we will use the 4×2-dimensional projection matrix W to transform our samples onto the new subspace via the equation Y=X×W, where the output matrix Y will be a 150×2 matrix of our transformed samples.



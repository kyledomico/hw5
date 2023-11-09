import numpy as np
import matplotlib.pyplot as plt
# Implement a function that performs PCA on a dataset without subtracting the mean and without normalizing the data and without scikitlearn.
# Return the d dimensional representations, estimated parameters, and reconstrution of these representations in D dimensions
def buggyPCA(data, d):
    # Perform SVD on the data matrix
    U, s, V = np.linalg.svd(data)

    # Take the first d columns of V as the principal components
    principal_components = V[:d].T

    # Get the d dimensional representations of the data
    d_dim_representations = np.dot(data, principal_components)

    # Get the estimated parameters
    estimated_parameters = np.diag(s[:d])

    # Get the reconstruction of the data
    reconstruction = np.dot(d_dim_representations, principal_components.T)

    return d_dim_representations, estimated_parameters, reconstruction

# Implement a function that performs PCA on a dataset that subtracts the mean but does not normalize the data and without scikitlearn.
# Return the d dimensional representations, estimated parameters, and reconstrution of these representations in D dimensions
def demeanedPCA(data, d):
    # Subtract the mean along each dimension of the nxD data matrix
    X = data - np.mean(data, axis=0)

    # Perform SVD on the data matrix
    U, s, V = np.linalg.svd(X)

    # Take the first d columns of V as the principal components
    principal_components = V[:d].T

    # Get the d dimensional representations of the data
    d_dim_representations = np.dot(X, principal_components)

    # Get the estimated parameters
    estimated_parameters = np.diag(s[:d])

    # Get the reconstruction of the data
    reconstruction = np.dot(d_dim_representations, principal_components.T) + np.mean(data, axis=0)

    return d_dim_representations, estimated_parameters, reconstruction

# Implement a function that performs PCA on a dataset that subtracts the mean and normalizes the data and without scikitlearn.
# Return the d dimensional representations, estimated parameters, and reconstrution of these representations in D dimensions
def normalizedPCA(data, d):
    # Subtract the mean along each dimension of the nxD data matrix
    X = data - np.mean(data, axis=0)

    # Normalize the data
    X = X / np.std(data, axis=0)

    # Perform SVD on the data matrix
    U, s, V = np.linalg.svd(X)

    # Take the first d columns of V as the principal components
    principal_components = V[:d].T

    # Get the d dimensional representations of the data
    d_dim_representations = np.dot(X, principal_components)

    # Get the estimated parameters
    estimated_parameters = np.diag(s[:d])

    # Get the reconstruction of the data
    reconstruction = np.dot(d_dim_representations, principal_components.T) * np.std(data, axis=0) + np.mean(data, axis=0)

    return d_dim_representations, estimated_parameters, reconstruction

# Implement the DRO Algorithm
# Return the d dimensional representations, estimated parameters, and reconstrution of these representations in D dimensions
def DRO(data, d):
    # Compute x_bar to be used for b
    x_bar = np.mean(data, axis=0)
    b = np.expand_dims(x_bar, axis=1)

    one_bT = np.ones((data.shape[0], 1)) @ b.transpose()

    # Perform SVD on the X - one_bT matrix
    U, s, V = np.linalg.svd(data - one_bT)

    # Take the first d columns of V as the principal components and set its transpose equal to A
    A = V[:d].T

    # Take the first d columns of U and the first d rows and d columns of s, matrix multiply them, and set the result equal to Z
    Z = U[:, :d] @ np.diag(np.diag(s[:d]))

    # Get the d dimensional representations of the data
    d_dim_representations = (data - one_bT) @ A

    # Get the estimated parameters
    estimated_parameters = np.diag(s[:d])

    # Get the reconstruction of the data
    reconstruction = d_dim_representations @ A.T + one_bT

    return d_dim_representations, estimated_parameters, reconstruction

# Read in the data2D.csv file as a numpy array
data = np.genfromtxt('data2D.csv', delimiter=',')

# Read in the data1000D.csv file as a numpy array
data1000D = np.genfromtxt('data1000D.csv', delimiter=',')

# Perform all 4 methods on the data2D numpy array, plot the results of each method in separate figures, and save the figures
results = [buggyPCA(data, 1), demeanedPCA(data, 1), normalizedPCA(data, 1), DRO(data, 1)]
labels = ['buggyPCA', 'demeanedPCA', 'normalizedPCA', 'DRO']
for i, result in enumerate(results):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], label='Original Data')
    plt.scatter(result[2][:, 0], result[2][:, 1], label='Reconstruction')
    plt.legend()
    plt.title(labels[i])
    plt.savefig(labels[i] + '2D.pdf')
    plt.clf()

# Write a function that compute the reconstruction error
def reconstructionError(data, reconstruction):
    total = 0
    for n in range(data.shape[0]):
        total += np.linalg.norm(data[n] - reconstruction[n])
    return total

# Print the reconstruction error of each method and the estimated parameters of each method
for i, result in enumerate(results):
    print(labels[i] + ' Reconstruction Error: ' + str(reconstructionError(data, result[2])) + "\\\\")
    print(labels[i] + ' Estimated Parameters: ' + str(result[1][0][0]) + "\\\\")

# Perform DRO on the data1000D numpy array, plot the siingular values of the estimated parameters, and save the figure
result = DRO(data1000D, 1000)
plt.figure()
plt.plot([i+1 for i in range(result[1].shape[0])], list(np.diag(result[1])))
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.title('DRO Singular Values on data1000D')
plt.savefig('DRO_singular_values1000D.pdf')
plt.clf()

# Plot an Information Retention Graph
svs = [i+1 for i in range(result[1].shape[0])]
cumulative_sum = sum(list(np.diag(result[1])))
cumulative_sums = []
recurring = list(np.diag(result[1]))[0]
cumulative_sums.append(recurring/cumulative_sum)
for sv in list(np.diag(result[1]))[1:]:
    recurring += sv
    cumulative_sums.append(recurring/cumulative_sum)  

plt.figure()
plt.plot(svs, cumulative_sums)
plt.xlabel('Singular Value Index')
plt.ylabel('Cumulative Sum of Singular Values')
plt.title('Information Retention Graph')
plt.savefig('information_retention_graph.pdf')
plt.clf()

# Perform all 4 methods on the data1000D numpy array, report the reconstruction error of each method
results = [buggyPCA(data1000D, 30), demeanedPCA(data1000D, 30), normalizedPCA(data1000D, 30), DRO(data1000D, 30)]

for i, result in enumerate(results):
    print(labels[i] + ' Reconstruction Error: ' + str(reconstructionError(data1000D, result[2])) + "\\\\")
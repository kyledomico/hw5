import numpy as np
from scipy.stats import multivariate_normal
# Create a two-dimensional synthetic dataset of 300 points by sampling 100 points each from the three Gaussian distributions shown below:
# P_a = N([-1, -1], \sigma [[2, 0.5], [0.5, 1]])
# P_b = N([1, -1], \sigma [[1, -0.5], [-0.5, 1]])
# P_c = N([0, 1], \sigma [[1, 0], [0, 2]])
# Where \sigma is a parameter we can change to produce different results

# Function that will return a 2D array of 300 points sampled from the three Gaussian distributions with parameter sigma
def generateSamples(sigma):
    # Create the three Gaussian distributions
    mean_a = [-1, -1]
    cov_a = [[2, 0.5], [0.5, 1]]
    mean_b = [1, -1]
    cov_b = [[1, -0.5], [-0.5, 1]]
    mean_c = [0, 1]
    cov_c = [[1, 0], [0, 2]]

    # Multiply the covariance matrices by sigma
    cov_a = np.multiply(cov_a, sigma)
    cov_b = np.multiply(cov_b, sigma)
    cov_c = np.multiply(cov_c, sigma)

    # Create the three Gaussian distributions
    dist_a = multivariate_normal(mean_a, cov_a)
    dist_b = multivariate_normal(mean_b, cov_b)
    dist_c = multivariate_normal(mean_c, cov_c)

    # Sample 100 points from each Gaussian distribution
    samples_a = dist_a.rvs(100)
    samples_b = dist_b.rvs(100)
    samples_c = dist_c.rvs(100)

    # Combine the three sets of samples into one array
    samples = np.concatenate((samples_a, samples_b, samples_c), axis=0)

    # Create an array of labels from which distribution each point was sampled from
    labels = np.concatenate((np.zeros(100), np.ones(100), np.full(100, 2)), axis=0)

    return samples, labels

# Implement the K-means++ algorithm as a function that takes as input the number of clusters K and the dataset X, and returns the cluster centers
def kmeanspp(K, X):
    # Initialize the cluster centers list
    centers = []

    # Randomly select the first cluster center from the dataset
    first_center = np.random.choice(X.shape[0], 1, replace=False)
    centers.append(X[first_center])

    # Loop through the rest of the cluster centers
    for i in range(1, K):
        # Create a list to store the distances of each point to the nearest cluster center
        distances = []

        # Loop through each point in the dataset
        for point in X:
            # Initialize the minimum distance to infinity
            min_dist = np.inf

            # Loop through each cluster center
            for center in centers:
                # Calculate the distance between the point and the cluster center
                dist = np.linalg.norm(point - center)

                # If the distance is less than the minimum distance, update the minimum distance
                if dist < min_dist:
                    min_dist = dist

            # Add the minimum distance to the list of distances
            distances.append(min_dist)

        # Convert the list of distances to a numpy array
        distances = np.array(distances)

        # Calculate the probability of each point being selected as the next cluster center
        probs = distances / np.sum(distances)

        # Randomly select the next cluster center from the dataset
        next_center = np.random.choice(X.shape[0], 1, replace=False, p=probs)
        centers.append(X[next_center])

    # Convert the list of cluster centers to a numpy array
    centers = np.array(centers)

    # Perform k-means clustering with the cluster centers and loop until the assignments to data points don't change
    while True:
        # Initialize the list of cluster assignments
        assignments = []

        # Loop through each point in the dataset
        for point in X:
            # Initialize the minimum distance to infinity
            min_dist = np.inf

            # Initialize the cluster assignment to -1
            assignment = -1

            # Loop through each cluster center
            for i, center in enumerate(centers):
                # Calculate the squared distance between the point and the cluster center
                dist = np.linalg.norm(point - center) ** 2

                # If the distance is less than the minimum distance, update the minimum distance and cluster assignment
                if dist < min_dist:
                    min_dist = dist
                    assignment = i

            # Add the cluster assignment to the list of assignments
            assignments.append(assignment)

        # Convert the list of assignments to a numpy array
        assignments = np.array(assignments)

        # Initialize a list to store the new cluster centers
        new_centers = []

        # Loop through each cluster center
        for i in range(K):
            # Create a list to store the points assigned to the cluster
            cluster = []

            # Loop through each point in the dataset
            for j, point in enumerate(X):
                # If the point is assigned to the cluster, add the point to the cluster
                if assignments[j] == i:
                    cluster.append(point)
            
            # Convert the list of points to a numpy array
            cluster = np.array(cluster)

            # Calculate the mean point of the cluster
            mean = np.mean(cluster, axis=0)
            
            # Add the mean point to the list of new cluster centers
            new_centers.append(mean)
        
        # Check to see if the cluster centers have changed
        if np.array_equal(centers, new_centers):
            break
        else:
            centers = np.array(new_centers)

    return centers

# Implement a Gaussian Mixture Model witht the EM algorithm as a function that takes as input the number of clusters K and the dataset X, and returns the parameters of the model
def gmm(K, X):
    # Initialize the weights, means, and covariance matrices
    weights = np.ones(K) / K
    means = X[np.random.choice(X.shape[0], K, replace=False)]
    covariances = np.array([np.cov(X, rowvar=False)] * K)

    # Loop until the log-likelihood converges
    while True:
        prev_means = np.copy(means)

        # E-Step
        posteriors = np.zeros((X.shape[0], K))
        for i in range(K):
            likelihood = multivariate_normal.pdf(X, means[i], covariances[i])
            posteriors[:, i] = weights[i] * likelihood
        
        posteriors /= posteriors.sum(axis=1, keepdims=True)

        # M-Step
        means = np.dot(posteriors.T, X) / posteriors.sum(axis=0)[:, np.newaxis]

        for i in range(K):
            diff = X - means[i]
            covariances[i] = np.dot(posteriors[:, i] * diff.T, diff) / posteriors[:, i].sum()
        
        weights = posteriors.sum(axis=0) / X.shape[0]

        # Check to see if the log-likelihood has converged
        if np.all(np.abs(means - prev_means) < 0.001):
            break
        
    return weights, means, covariances

# Loop for different values of sigma [0.5, 1, 2, 4, 8]
sigmas = [0.5, 1, 2, 4, 8]
kmeans_objectives = []
gmm_objectives = []
kmeans_accuracies = []
gmm_accuracies = []
for sigma in sigmas:
    # Generate the data
    samples, labels = generateSamples(sigma)

    # Run K-means++ with K = 3 an evaluate the objective function
    centers = kmeanspp(3, samples)

    k_means_objective = 0
    for point in samples:
        # Initialize the minimum distance to infinity
        min_dist = np.inf

        # Loop through each cluster center
        for i, center in enumerate(centers):
            # Calculate the squared distance between the point and the cluster center
            dist = np.linalg.norm(point - center) ** 2

            # If the distance is less than the minimum distance, update the minimum distance and cluster assignment
            if dist < min_dist:
                min_dist = dist

        # Add the min distance to the objective function
        k_means_objective += min_dist
    kmeans_objectives.append(k_means_objective)

    # Run GMM with K = 3 and evaluate the negative log-likelihood
    weights, means, covariances = gmm(3, samples)

    gmm_objective = 0
    for i in range(3):
        gmm_objective += weights[i] * multivariate_normal.pdf(samples, means[i], covariances[i])
    gmm_objective = -np.log(gmm_objective).sum()
    gmm_objectives.append(gmm_objective)

    # Create data point assignments for K-means++ and GMM
    kmeans_assignments = []
    gmm_assignments = []
    for point in samples:
        # Initialize the minimum distance to infinity
        min_dist = np.inf

        # Initialize the cluster assignment to -1
        assignment = -1

        # Loop through each cluster center
        for i, center in enumerate(centers):
            # Calculate the squared distance between the point and the cluster center
            dist = np.linalg.norm(point - center) ** 2

            # If the distance is less than the minimum distance, update the minimum distance and cluster assignment
            if dist < min_dist:
                min_dist = dist
                assignment = i

        # Add the cluster assignment to the list of assignments
        kmeans_assignments.append(assignment)
    
    for point in samples:
        # Initialize the maximum probability to 0
        max_prob = 0

        # Initialize the cluster assignment to -1
        assignment = -1

        # Loop through each cluster center
        for i in range(3):
            # Calculate the probability of the point belonging to the cluster
            prob = weights[i] * multivariate_normal.pdf(point, means[i], covariances[i])

            # If the probability is greater than the maximum probability, update the maximum probability and cluster assignment
            if prob > max_prob:
                max_prob = prob
                assignment = i

        # Add the cluster assignment to the list of assignments
        gmm_assignments.append(assignment)
    
    # Compute the accuracy of K-means++ and GMM by comparing the data point assignments to the true labels
    kmeans_accuracy = 0
    gmm_accuracy = 0
    for i in range(300):
        if kmeans_assignments[i] == labels[i]:
            kmeans_accuracy += 1
        if gmm_assignments[i] == labels[i]:
            gmm_accuracy += 1
    kmeans_accuracy /= 300
    gmm_accuracy /= 300
    kmeans_accuracies.append(kmeans_accuracy)
    gmm_accuracies.append(gmm_accuracy)

# Plot the objective functions for K-means++ and GMM as a function of sigma
import matplotlib.pyplot as plt
plt.plot(sigmas, kmeans_objectives, label='K-means++')
plt.plot(sigmas, gmm_objectives, label='GMM')
plt.xlabel('Sigma')
plt.ylabel('Objective Function')
plt.title('Objective Function vs. Sigma')
plt.legend()
plt.savefig('problem1_2clusteringobjective.pdf')
plt.clf()

# Plot the accuracy of K-means++ and GMM as a function of sigma
plt.plot(sigmas, kmeans_accuracies, label='K-means++')
plt.plot(sigmas, gmm_accuracies, label='GMM')
plt.xlabel('Sigma')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Sigma')
plt.legend()
plt.savefig('problem1_2clusteringaccuracy.pdf')
plt.clf()



    


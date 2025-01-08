import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

def generate_3d_data(n_samples=100, random_seed=42):
    """
    Generate a random 3D dataset.
    
    Parameters:
        n_samples (int): Number of data points.
        random_seed (int): Seed for reproducibility.

    Returns:
        np.ndarray: A (n_samples, 3) array of random data.
    """
    np.random.seed(random_seed)
    return np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[1, 0.8, 0.6], [0.8, 1, 0.8], [0.6, 0.8, 1]],
        size=n_samples
    )

# Step 1: Generate the dataset
data = generate_3d_data()

# Step 2: Compute the covariance matrix of the original data
cov_matrix = np.cov(data.T)

# Step 3: Perform PCA
pca = PCA(n_components=3)
pca.fit(data)

# Step 4: Singular Vectors (Principal Components) and Singular Values (Eigenvalues)
singular_vectors = pca.components_
singular_values = pca.singular_values_

# Step 5: Visualize the covariance matrix, singular values (as a bar plot), and mapped singular vectors
fig = plt.figure(figsize=(18, 6))

# Subplot 1: Covariance Matrix
ax1 = fig.add_subplot(131)
sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt='.2f', ax=ax1, linewidths=0.5)
ax1.set_title("Covariance Matrix Heatmap")

# Subplot 2: Singular Values as Bar Plot (Bins)
ax2 = fig.add_subplot(132)
colors = ['r', 'g', 'b']  # Colors corresponding to the principal components
for i in range(3):
    ax2.bar(i+1, singular_values[i], color=colors[i], alpha=0.7, label=f"PC {i+1}")
ax2.set_title("Singular Values (Variance Explained)")
ax2.set_xlabel('Principal Component')
ax2.legend()

# Subplot 3: Mapping Singular Vectors on the Original Dataset
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', alpha=0.7, label='Original Data')

# Visualize each singular vector (without multiplying by singular values) on the original dataset
for i in range(3):
    ax3.quiver(np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2]), 
               singular_vectors[0, i], singular_vectors[1, i], singular_vectors[2, i], 
               color=colors[i], length=2, label=f"PC {i+1}")

ax3.set_title("Singular Vectors Mapped to Original Data")
ax3.legend()

plt.tight_layout()
plt.show()

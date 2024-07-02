import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate or load your 3D direction data
# For demonstration, let's generate random 3D direction data
def plot_data(data):

    # Step 2: Normalize the data to unit vectors (if not already normalized)
    norms = np.linalg.norm(data, axis=1)
    unit_vectors = data / norms[:, np.newaxis]

    # Step 3: Convert 3D coordinates to spherical coordinates
    theta = np.arccos(unit_vectors[:, 2])  # Polar angle
    phi = np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0])  # Azimuthal angle

    # Step 4: Create a histogram of the spherical coordinates
    # Number of bins for theta and phi
    num_bins = 20

    # Create 2D histogram in spherical coordinates
    hist, theta_edges, phi_edges = np.histogram2d(theta, phi, bins=num_bins, range=[[0, np.pi], [-np.pi, np.pi]])

    # Step 5: Plot arrows representing the density of directions in each bin
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    theta_mid = (theta_edges[:-1] + theta_edges[1:]) / 2
    phi_mid = (phi_edges[:-1] + phi_edges[1:]) / 2
    theta_grid, phi_grid = np.meshgrid(theta_mid, phi_mid, indexing='ij')

    # Calculate arrow directions and lengths
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    # Scale factor for arrow length based on histogram counts
    scale = hist / hist.max()

    # Plot arrows
    for i in range(num_bins):
        for j in range(num_bins):
            if hist[i, j] > 0:  # Plot only for non-empty bins
                ax.quiver(0, 0, 0, x[i, j] * scale[i, j], y[i, j] * scale[i, j], z[i, j] * scale[i, j],
                        length=scale[i, j] / 50, normalize=True, color=plt.cm.viridis(scale[i, j]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Histogram of 3D Directions with Arrows')

    plt.show()

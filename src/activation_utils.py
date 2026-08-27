"""Activation utils.

This module contains functions to compute activation ranges and
bitmaps of activations.
"""

from typing import List, Tuple

import torch
import numpy as np

from src import vecquantile
from src import constants as C


def import_algorithm(algorithm: str) -> None:
    """Import the clustering algorithm.

    Args:
        algorithm (str): Algorithm to use for clustering.

    Returns:
        None
    """
    if algorithm == "kmeans":
        from sklearn.cluster import KMeans
    elif algorithm == "gaussian_mixture":
        from sklearn.mixture import GaussianMixture
    elif algorithm == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering
    elif algorithm == "hdbscan":
        from hdbscan import HDBSCAN
    else:
        raise ValueError(
            "Unknown clustering algorithm. "
            "Values: [kmeans, gaussian_mixture, hierarchical, hdbscan]"
        )


def build_ranges_from_clusters(
        activations: torch.Tensor, clusters: List[int],
        num_clusters: int) -> List[tuple]:
    """Build activation ranges from clusters.

    Args:
        activations (torch.Tensor): Activations of the unit.
        clusters (List[int]): Clusters indexes of the activations.
        num_clusters (int): Number of clusters.

    Returns:
        activation_ranges (List[tuple]): Activation ranges for each cluster.
    """
    if num_clusters == -1:
        cluster_labels = torch.unique(clusters)
        cluster_labels = cluster_labels[cluster_labels != -1].tolist()
    else:
        cluster_labels = list(range(num_clusters))

    cluster_labels = sorted(cluster_labels, key=lambda l: activations[clusters == l].min().item())

    activations_ranges = []

    for label in cluster_labels:
        cluster_activations = activations[clusters == label]
        lower_bound = torch.min(cluster_activations).item()
        upper_bound = torch.max(cluster_activations).item()
        activations_ranges.append((lower_bound, upper_bound))

    min_bound = torch.min(activations[clusters != -1]).item()
    max_bound = torch.max(activations[clusters != -1]).item()

    if torch.any(activations < min_bound):
        print("Prepending artificial cluster for noisy low activations")
        activations_ranges.insert(0, (float("-inf"), min_bound))

    if torch.any(activations > max_bound):
        print("Appending artificial cluster for noisy high activations")
        activations_ranges.append((max_bound, float("inf")))

    return activations_ranges


def compute_activation_ranges(
        activations: torch.Tensor, num_clusters: int, algorithm: str) -> List[Tuple]:
    """Compute activation ranges for each unit.

    Args:
        activations (torch.Tensor): Activations of the unit.
        num_clusters (int): Number of clusters.
        algorithm (str): Algorithm to use for clustering.

    Returns:
        activation_ranges (List[tuple]): Activation ranges for each unit.
    """
    if num_clusters == 1:
        # Case vanilla compositional and netdissect range
        # Avoid zero is set to false like in the compositional paper
        print("Skipping clustering because num_clusters is 1")
        threshold = quantile_threshold(
            activations, quantile=C.NETDISSECT_QUANTILE, avoid_zero=False
        )
        return [(threshold, torch.tensor(float("inf")))]

    import_algorithm(algorithm)
    
    activations = activations.reshape(-1, 1)
    # Remove zeros from activations if there is a relu activation
    if torch.all(activations >= 0):
        activations = activations[activations > 0]
        activations = activations.reshape(-1, 1)

    np.random.seed(0)

    # Compute activation ranges
    if algorithm == "kmeans":
        from sklearn.cluster import KMeans
        print("Running K-Means clustering")
        clusters = KMeans(
            n_clusters=num_clusters, random_state=0, n_init=1
        ).fit_predict(activations.detach().cpu().numpy())

    elif algorithm == "gaussian_mixture":
        from sklearn.mixture import GaussianMixture
        print("Running Gaussian Mixture Model")
        gmm = GaussianMixture(n_components=num_clusters, random_state=0)
        clusters = gmm.fit_predict(activations.detach().cpu().numpy())

    elif algorithm == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering
        print("Running Hierarchical clustering")
        clusters = AgglomerativeClustering(n_clusters=num_clusters).fit_predict(activations.detach().cpu().numpy())

    elif algorithm == "hdbscan":
        from hdbscan import HDBSCAN
        print("Running HDBSCAN")
        print("Ignoring num_clusters flag if set; HDBSCAN automatically determines the number of clusters")
        num_clusters = -1
        clusters = HDBSCAN(
            min_cluster_size=int(0.02 * activations.numel()), min_samples=50
        ).fit_predict(activations.detach().cpu().numpy())
        clusters = torch.tensor(clusters)

    else:
        raise ValueError("Unknown clustering algorithm. Values: [kmeans, gaussian_mixture, hierarchical, hdbscan]")

    activation_ranges = build_ranges_from_clusters(activations, clusters, num_clusters)

    return activation_ranges


def compute_bitmaps(
        activations: torch.Tensor, activation_range: Tuple,
        mask_shape: List[int]) -> torch.Tensor:
    """Get the bitmaps of the unit.

    This function upsamples the activations to the original size of the
    image and then binarize them.
    Args:
        activations (torch.Tensor): Activations of the unit.
        activation_range (Tuple): Activation range of the unit.
        mask_shape (List[int]): Shape of the mask.

    Returns:
        bitmaps (torch.Tensor): Bitmaps of the unit.
    """
    lower, upper = activation_range
    upsampled_activations = torch.nn.functional.interpolate(
        activations.unsqueeze(1),
        size=mask_shape, mode='bilinear')
    upsampled_activations = upsampled_activations.squeeze(1)
    bitmaps = torch.where(
        (upsampled_activations > lower) & (upsampled_activations <= upper),
        True, False)
    bitmaps = bitmaps.reshape(bitmaps.shape[0], -1)
    return bitmaps


def quantile_threshold(
        layer_activations: torch.Tensor, quantile: float, *,
        avoid_zero: bool, batch_size=64, seed=1) -> torch.Tensor:
    """
    Determine thresholds for neuron activations for each neuron.

    Args:
        layer_activations (torch.Tensor): Activations of the layer.
        quantile (float): Quantile to use.
        avoid_zero (bool): Whether to remove zeros from the activations.
        batch_size (int): Batch size to use.
        seed (int): Seed to use for the quantile vector.

    Returns:
        thresholds (torch.Tensor): Thresholds for each neuron.
    """
    quant = vecquantile.QuantileVector(depth=1, seed=seed)
    for i in range(0, layer_activations.shape[0], batch_size):
        batch = layer_activations[i:i + batch_size]
        batch = batch.flatten().reshape(-1, 1)
        if avoid_zero:
            batch = batch[batch != 0].reshape(-1, 1)
        quant.add(batch)
    thresholds = quant.readout(1000)[:, int(1000 * (1 - quantile) - 1)]
    return torch.tensor(thresholds)

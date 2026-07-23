"""Activation utils.

This module contains functions to compute activation ranges and
bitmaps of activations.
"""

from typing import List, Tuple

import sys
import torch
import numpy as np
from hdbscan import HDBSCAN

from src import vecquantile
from src import constants as C


def build_ranges_from_clusters(
        activations: torch.Tensor, 
        clusters: torch.Tensor,
    ) -> List[tuple]:

    unique_labels = torch.unique(clusters)
    unique_labels = unique_labels[unique_labels != -1].tolist()
    unique_labels = sorted(unique_labels, key=lambda l: activations[clusters == l].min().item())

    activations_ranges = []
    print(f"Activations: {activations.numel()}")
    print(f"Number of Clusters: {len(unique_labels) + 1}")
    if (len(unique_labels)) > 100:
        sys.exit("Error: Too many clusters")
    print(f"Overall Noise: {100 * activations[clusters == -1].numel() / activations.numel():.2f}%")
    
    upper_bound = None
    i = 0
    for label in unique_labels:
        print(f"\nCluster {i}:")
        cluster_activations = activations[clusters == label]
        lower_bound = torch.min(cluster_activations).item()
        upper_bound = torch.max(cluster_activations).item()
        print(f"({lower_bound:.3f}, {upper_bound:.3f})")
        print(
            "Size: "
            f"{100 * activations[(lower_bound <= activations) & (activations <= upper_bound)].numel() / activations.numel():.2f}%"
            )
        noise_density = (
            activations[(clusters == -1) & (lower_bound <= activations.flatten()) & (activations.flatten() <= upper_bound)].numel()
            / activations[(lower_bound <= activations) & (activations <= upper_bound)].numel()
        )
        print(f"Noise: {100 * noise_density:.2f}%")
        activations_ranges.append((lower_bound, upper_bound))
        i += 1

    if upper_bound is not None:
        print(f"\nCluster {i} (artificial):")
        print(f"({upper_bound:.3f}, inf)")
        print(f"Size: {100 * activations[activations > upper_bound].numel() / activations.numel():.2f}%\n")
        print(f"Noise: 100%")
        activations_ranges.append((upper_bound, float("inf")))

    return activations_ranges


def compute_activation_ranges(
        activations: torch.Tensor, num_clusters: int) -> List[Tuple]:
    """Compute activation ranges for each unit.

    Args:
        activations (torch.Tensor): Activations of the unit.
        num_clusters (int): Number of clusters (ignored).

    Returns:
        activation_ranges (List[tuple]): Activation ranges for each unit.
    """
    if num_clusters != -1:
        print("Ignoring num_clusters flag because of non-fixed clustering")
        
    activations = activations.reshape(-1, 1)
    
    # Remove zeros from activations if there is a relu activation
    if torch.all(activations >= 0):
        activations = activations[activations > 0]
        activations = activations.reshape(-1, 1)

    # Compute activation ranges
    np.random.seed(0)
    clusters = HDBSCAN(
        min_cluster_size=int(0.02 * activations.numel()),
        min_samples=50
    ).fit_predict(activations.detach().cpu().numpy())

    activation_ranges = build_ranges_from_clusters(activations, torch.tensor(clusters))
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
        (upsampled_activations > lower) & (upsampled_activations < upper),
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
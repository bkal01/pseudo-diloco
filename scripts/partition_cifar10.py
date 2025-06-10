#!/usr/bin/env python3

import argparse
import os
import pickle
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.pseudo_diloco.partitioning.embeddings import embed, cluster


def visualize_embeddings(embeddings, labels, clusters, save_dir):
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())
    
    cluster_assignments = np.zeros(len(embeddings))
    for cluster_id, cluster_indices in enumerate(clusters):
        cluster_assignments[cluster_indices] = cluster_id
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot colored by clusters
    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=cluster_assignments, cmap='tab10', alpha=0.7, s=10)
    ax1.set_title('Embeddings Colored by Clusters')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
    
    # Plot colored by true labels
    scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels.numpy(), cmap='tab10', alpha=0.7, s=10)
    ax2.set_title('Embeddings Colored by True Labels')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter2, ax=ax2, label='CIFAR-10 Class')
    
    plt.tight_layout()
    
    vis_path = os.path.join(save_dir, f'embeddings_visualization_{len(clusters)}.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    
    print(f"Visualization saved to {vis_path}")


def main():
    parser = argparse.ArgumentParser(description='Partition CIFAR-10 using embeddings and clustering')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save clustered indices')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of samples in validation set (default: 10000)')
    parser.add_argument('--num_clusters', type=int, default=2,
                        help='Number of clusters to create (default: 2)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--vis', action='store_true',
                        help='Create t-SNE visualization of embeddings')
    
    args = parser.parse_args()
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10(
        root='data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Split dataset
    full_size = len(dataset)
    train_size = full_size - args.val_size
    
    indices = torch.randperm(full_size).tolist()
    train_indices = indices[:train_size]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Creating embeddings for training set...")
    train_embeddings, train_labels = embed(train_loader)
    print(f"Train embeddings shape: {train_embeddings.shape}")
    
    print(f"Clustering training embeddings into {args.num_clusters} clusters...")
    train_clusters = cluster(train_embeddings, args.num_clusters)
    
    # Convert cluster indices back to original dataset indices
    train_clusters_original = []
    for cluster_indices in train_clusters:
        original_indices = [train_indices[i] for i in cluster_indices]
        train_clusters_original.append(np.array(original_indices))
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    cluster_file = os.path.join(args.save_dir, f"train_clusters_{args.num_clusters}.pkl")
    with open(cluster_file, 'wb') as f:
        pickle.dump(train_clusters_original, f)
    
    labels_file = os.path.join(args.save_dir, f"train_labels_{args.num_clusters}.pkl")
    with open(labels_file, 'wb') as f:
        pickle.dump(train_labels.numpy(), f)
    
    # Print cluster statistics
    for i, cluster_indices in enumerate(train_clusters_original):
        cluster_labels = train_labels.numpy()[np.isin(train_indices, cluster_indices)]
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"Cluster {i}: {len(cluster_indices)} samples, "
              f"label distribution: {dict(zip(unique_labels, counts))}")
    
    # Create visualization if requested
    if args.vis:
        visualize_embeddings(train_embeddings, train_labels, train_clusters, args.save_dir)


if __name__ == "__main__":
    main() 
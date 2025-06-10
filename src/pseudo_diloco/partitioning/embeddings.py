import numpy as np
import torch

from sklearn.cluster import KMeans
from transformers import ImageGPTImageProcessor, ImageGPTModel
from tqdm import tqdm
from typing import Tuple, List

device = "cuda" if torch.cuda.is_available() else "cpu"

def embed(dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
    processor = ImageGPTImageProcessor.from_pretrained("openai/imagegpt-small")
    model = ImageGPTModel.from_pretrained("openai/imagegpt-small")
    model.to(device)
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            
            inputs = processor(images=images, do_normalize=True, return_tensors="pt")
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            
            embeddings = embeddings.mean(dim=1)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    return embeddings_tensor, labels_tensor


def cluster(embeddings: torch.Tensor, k: int) -> List[np.ndarray]:
    embeddings_np = embeddings.numpy()
    
    kmeans = KMeans(n_clusters=k, init="random", max_iter=500)
    cluster_labels = kmeans.fit_predict(embeddings_np)
    
    clusters = []
    for cluster_id in range(k):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        clusters.append(cluster_indices)
    
    return clusters 
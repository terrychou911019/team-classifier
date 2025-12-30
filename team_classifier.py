import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from typing import Generator, Iterable, List, TypeVar
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoProcessor, SiglipVisionModel
from loguru import logger

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
        Initialize the TeamClassifier.

        Args:
            device (str): Computation device ('cpu' or 'cuda').
            batch_size (int): Batch size for feature extraction.
        """
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading SigLIP model on {self.device}...")
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        
        # Initialize UMAP and KMeans
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2, n_init='auto') 

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using SigLIP.

        Args:
            crops (List[np.ndarray]): List of image crops in BGR format (OpenCV default).

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        # Convert OpenCV (BGR) -> Pillow (RGB) because HuggingFace models expect RGB PIL images
        pil_crops = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops]
        
        batches = create_batches(pil_crops, self.batch_size)
        data = []
        
        with torch.inference_mode(): 
            for batch in batches:
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                
                # Use the mean of the last hidden state as the feature embedding
                # Shape: [batch_size, embedding_dim]
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        if not data:
            return np.array([])
        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops (Training phase).
        
        Args:
            crops (List[np.ndarray]): List of representative image crops.
        """
        if not crops:
            logger.warning("No crops provided for fitting.")
            return
        
        logger.info(f"Extracting features for fitting ({len(crops)} samples)...")
        data = self.extract_features(crops)
        
        logger.info("Running UMAP reduction...")
        projections = self.reducer.fit_transform(data)
        
        logger.info("Running K-Means clustering...")
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels (Team IDs) for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops to classify.

        Returns:
            np.ndarray: Predicted cluster labels (0 or 1).
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)

        return self.cluster_model.predict(projections)
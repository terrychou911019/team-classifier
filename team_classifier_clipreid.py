import numpy as np
import torch
import cv2
from PIL import Image
from typing import Generator, Iterable, List, TypeVar
from sklearn.cluster import KMeans
from loguru import logger
import torchvision.transforms as T
from clipreid import model

V = TypeVar("V")

class ClIPReIDExtractor:
    """
    Feature extractor with CLIP-ReID model.
    """
    def __init__(self, model, device='cuda'):
        self.model = model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, batch_tensor):
        batch_tensor = batch_tensor.to(self.device, non_blocking=True)
        feats = self.model(batch_tensor) 
        return feats

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
    A classifier that uses a pre-trained ReID model for feature extraction,
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
        
        logger.info(f"Loading CLIP-ReID model on {self.device}...")
        
        self.features_model = ClIPReIDExtractor(model=model, device=device)

        self.preprocess = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.cluster_model = KMeans(n_clusters=2, n_init='auto') 

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using OSNet.

        Args:
            crops (List[np.ndarray]): List of image crops in BGR format (OpenCV default).

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        # Convert OpenCV (BGR) -> Pillow (RGB)
        pil_crops = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops]
        
        batches = create_batches(pil_crops, self.batch_size)
        data = []
        
        with torch.inference_mode(): 
            for batch in batches:
                batch_tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
                features = self.features_model(batch_tensors)
                data.append(features.cpu().numpy())

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
        
        logger.info("Running K-Means clustering...")
        self.cluster_model.fit(data)

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

        return self.cluster_model.predict(data)
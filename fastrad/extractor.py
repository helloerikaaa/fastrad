from .settings import FeatureSettings
from .image import MedicalImage, Mask

class FeatureExtractor:
    def __init__(self, settings: FeatureSettings):
        self.settings = settings
        # Device resolution logic goes here
        self.device = "cpu"

    def extract(self, image: MedicalImage, mask: Mask) -> dict[str, float]:
        features = {}
        # Computation loop goes here
        return features

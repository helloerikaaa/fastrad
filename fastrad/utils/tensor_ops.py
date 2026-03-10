import torch

def bin_image(image: torch.Tensor, bin_width: float) -> torch.Tensor:
    return torch.floor(image / bin_width)

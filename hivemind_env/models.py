import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for HiveMind:
    - Passes the 15x15x5 egocentric grid through a CNN (local spatial/obstacle perception).
    - Concatenates `is_carrying` scalar (one-hot Discrete(2) = 2 dims).
    - Concatenates `last_action` scalar (one-hot Discrete(7) = 7 dims).
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # CNN for the grid (15x15x5)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 15x15 -> 7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 7x7 -> 3x3
            nn.Flatten()
        )
        
        # Compute shape by doing one forward pass with dummy data
        with torch.no_grad():
            dummy_grid = torch.zeros(1, 5, 15, 15)
            cnn_output_dim = self.cnn(dummy_grid).shape[1]
            # SB3 automatically one-hot encodes Discrete(2) into 2 dims and Discrete(7) into 7 dims
            dummy_carrying = torch.zeros(1, 2)
            dummy_last_action = torch.zeros(1, 7)
            dummy_combined = torch.cat((torch.zeros(1, cnn_output_dim), dummy_carrying, dummy_last_action), dim=1)
            total_concat_dim = dummy_combined.shape[1]
        
        # Final linear layer to project to requested `features_dim`
        self.linear = nn.Sequential(
            nn.Linear(total_concat_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # 1. Process the Grid
        grid = observations["grid"]
        if grid.shape[-1] == 5:
            grid = grid.permute(0, 3, 1, 2)
            
        cnn_features = self.cnn(grid)
        
        # 2. Process is_carrying flag
        is_carrying = observations["is_carrying"].float()
        is_carrying = is_carrying.view(is_carrying.shape[0], -1)

        # 3. Process last_action flag
        last_action = observations["last_action"].float()
        last_action = last_action.view(last_action.shape[0], -1)
            
        # 4. Concatenate all features (576 + 2 + 7 = 585 dims)
        combined = torch.cat((cnn_features, is_carrying, last_action), dim=1)
        
        # 5. Project to features_dim
        return self.linear(combined)

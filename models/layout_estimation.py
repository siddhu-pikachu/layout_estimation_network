import torch
import torch.nn as nn
import torchvision.models as models


class LayoutEstimationNetwork(nn.Module):
    """
    Layout Estimation Network (LEN) for predicting 3D room layout from single images.
    Based on the approach described in the Total3D paper.

    The network uses a CNN backbone (ResNet) to extract features from the input image,
    followed by fully connected layers to predict layout parameters and camera parameters.

    Layout parameters:
    - layout_centroid (3) - x, y, z coordinates of the room center
    - layout_size (3) - width, height, depth of the room
    - layout_orientation (1) - orientation angle of the room

    Camera parameters:
    - camera_pitch (1) - pitch angle of the camera
    - camera_roll (1) - roll angle of the camera
    """

    def __init__(self, backbone='resnet18', pretrained=True, dropout=0.5):
        """
        Initialize the Layout Estimation Network.

        Args:
            backbone (str): The CNN backbone to use ('resnet18', 'resnet34', 'resnet50')
            pretrained (bool): Whether to use pretrained weights
            dropout (float): Dropout probability for fully connected layers
        """
        super(LayoutEstimationNetwork, self).__init__()

        # Load the appropriate backbone CNN
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Use all layers except the final FC layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # Layout parameter estimation
        self.fc_layout = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 7)  # 3 (centroid) + 3 (size) + 1 (orientation)
        )

        # Camera parameter estimation
        self.fc_camera = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)  # pitch and roll
        )

    def forward(self, x):
        """
        Forward pass of the Layout Estimation Network.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W]

        Returns:
            tuple: (layout_params, camera_params)
                - layout_params (torch.Tensor): Predicted layout parameters [B, 7]
                - camera_params (torch.Tensor): Predicted camera parameters [B, 2]
        """
        # Extract features using the backbone
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # Flatten the features

        # Predict layout and camera parameters
        layout_params = self.fc_layout(x)
        camera_params = self.fc_camera(x)

        return layout_params, camera_params

    def get_layout_components(self, layout_params):
        """
        Extract the individual components from the layout parameters.

        Args:
            layout_params (torch.Tensor): Layout parameters tensor [B, 7]

        Returns:
            tuple: (centroid, size, orientation)
                - centroid (torch.Tensor): Room centroid coordinates [B, 3]
                - size (torch.Tensor): Room size dimensions [B, 3]
                - orientation (torch.Tensor): Room orientation angle [B, 1]
        """
        centroid = layout_params[:, :3]  # x, y, z
        size = layout_params[:, 3:6]  # width, height, depth
        orientation = layout_params[:, 6:7]  # angle

        return centroid, size, orientation
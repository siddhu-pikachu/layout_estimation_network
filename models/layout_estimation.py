import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math


class LayoutEstimationNetwork(nn.Module):
    """
    Layout Estimation Network (LEN) for predicting 3D room layout from single images.
    Aligned with the Total3DUnderstanding paper's approach.

    The network uses a CNN backbone (ResNet) to extract features from the input image,
    followed by more sophisticated fully connected layers to predict layout parameters
    and camera parameters.

    Layout parameters:
    - layout_centroid (3) - x, y, z coordinates of the room center
    - layout_size (3) - width, height, depth of the room (strictly positive)
    - layout_orientation (1) - orientation angle of the room

    Camera parameters:
    - camera_pitch (1) - pitch angle of the camera
    - camera_roll (1) - roll angle of the camera
    """

    def __init__(self, backbone='resnet34', pretrained=True):
        """
        Initialize the Layout Estimation Network.

        Args:
            backbone (str): The CNN backbone to use ('resnet18', 'resnet34', 'resnet50')
            pretrained (bool): Whether to use pretrained weights
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

        # Layout feature extraction branch
        self.layout_encoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        # Separate regressors for each layout component for better learning
        self.centroid_regressor = nn.Linear(512, 3)  # x, y, z
        self.size_regressor = nn.Linear(512, 3)  # width, height, depth
        self.orientation_regressor = nn.Linear(512, 1)  # single angle

        # Camera parameter estimation network
        self.camera_encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.camera_regressor = nn.Linear(512, 2)  # pitch and roll

        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = torch.flatten(x, 1)  # Flatten the features [B, feature_dim]

        # Layout parameters prediction
        layout_features = self.layout_encoder(x)

        # Predict centroid (unrestricted)
        centroid = self.centroid_regressor(layout_features)

        # Predict size (strictly positive using exponential)
        size = torch.exp(self.size_regressor(layout_features))

        # Predict orientation angle
        orientation = self.orientation_regressor(layout_features)

        # Predict camera parameters
        camera_features = self.camera_encoder(x)
        camera_params = self.camera_regressor(camera_features)

        # Combine layout parameters
        layout_params = torch.cat([centroid, size, orientation], dim=1)

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


class JointLayoutEstimationNetwork(nn.Module):
    """
    Enhanced LEN with context-aware reasoning for better scene understanding.
    This is an advanced version aligned with more recent approaches in the paper.

    The network adds a scene graph-based context module to refine the initial
    layout predictions by considering relationships and spatial constraints.
    """

    def __init__(self, backbone='resnet34', pretrained=True):
        """
        Initialize the Joint Layout Estimation Network.

        Args:
            backbone (str): The CNN backbone to use
            pretrained (bool): Whether to use pretrained weights
        """
        super(JointLayoutEstimationNetwork, self).__init__()

        # Base layout estimation network for initial predictions
        self.base_network = LayoutEstimationNetwork(backbone, pretrained)

        # Feature dimensions
        if backbone in ['resnet18', 'resnet34']:
            feature_dim = 512
        else:  # resnet50
            feature_dim = 2048

        # Context-aware refinement module
        self.context_encoder = nn.Sequential(
            nn.Linear(7 + 2 + feature_dim, 1024),  # layout params + camera params + image features
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        # Refinement prediction heads
        self.refine_centroid = nn.Linear(512, 3)
        self.refine_size = nn.Linear(512, 3)
        self.refine_orientation = nn.Linear(512, 1)
        self.refine_camera = nn.Linear(512, 2)

        # Initialize weights
        self._initialize_refinement_weights()

    def _initialize_refinement_weights(self):
        """Initialize refinement module weights."""
        for m in [self.context_encoder, self.refine_centroid,
                  self.refine_size, self.refine_orientation, self.refine_camera]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
                    elif isinstance(layer, nn.BatchNorm1d):
                        nn.init.constant_(layer.weight, 1)
                        nn.init.constant_(layer.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the Joint Layout Estimation Network.

        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]

        Returns:
            tuple: (refined_layout_params, refined_camera_params, initial_layout_params, initial_camera_params)
        """
        # Get backbone features
        features = self.base_network.backbone(x)
        features_flat = torch.flatten(features, 1)

        # Get initial predictions
        initial_layout, initial_camera = self.base_network(x)

        # Context-aware refinement
        context_input = torch.cat([initial_layout, initial_camera, features_flat], dim=1)
        context_features = self.context_encoder(context_input)

        # Predict refinements
        centroid_offset = self.refine_centroid(context_features)
        size_factor = torch.sigmoid(self.refine_size(context_features)) * 2.0  # Scale factor between 0-2
        orientation_offset = self.refine_orientation(context_features)
        camera_offset = self.refine_camera(context_features)

        # Apply refinements
        refined_centroid = initial_layout[:, :3] + centroid_offset
        refined_size = initial_layout[:, 3:6] * size_factor
        refined_orientation = initial_layout[:, 6:7] + orientation_offset
        refined_camera = initial_camera + camera_offset

        # Combine refined parameters
        refined_layout = torch.cat([refined_centroid, refined_size, refined_orientation], dim=1)

        return refined_layout, refined_camera, initial_layout, initial_camera

    def get_layout_components(self, layout_params):
        """
        Extract the individual components from the layout parameters.

        Args:
            layout_params (torch.Tensor): Layout parameters tensor [B, 7]

        Returns:
            tuple: (centroid, size, orientation)
        """
        return self.base_network.get_layout_components(layout_params)


# Use this model for the most advanced implementation
class Total3DLayoutEstimationNetwork(nn.Module):
    """
    Implementation that closely follows the Total3DUnderstanding paper.

    This version adds physical constraints and better handles size estimation
    with a dedicated representation for more stable training.
    """

    def __init__(self, backbone='resnet34', pretrained=True):
        """Initialize the Total3D Layout Estimation Network."""
        super(Total3DLayoutEstimationNetwork, self).__init__()

        # Image feature extractor
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

        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # Layout estimation branch
        self.layout_encoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        # Parameter-specific prediction heads as in the paper
        self.centroid_regressor = nn.Linear(512, 3)  # x, y, z

        # Size representation using coefficients and basis shapes
        # This helps stabilize the size prediction as mentioned in the paper
        self.basis_dim = 10
        self.size_basis = nn.Parameter(torch.randn(3, self.basis_dim))
        self.size_coeffs_regressor = nn.Linear(512, self.basis_dim)

        # Direct size regressor as a fallback
        self.size_regressor = nn.Linear(512, 3)

        # Orientation using classification and regression
        # This handles the cyclical nature of angles better
        self.orientation_bins = 24
        self.bin_size = 2 * np.pi / self.orientation_bins
        self.orientation_cls = nn.Linear(512, self.orientation_bins)
        self.orientation_reg = nn.Linear(512, self.orientation_bins)

        # Camera parameter estimation
        self.camera_encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Camera parameters (pitch and roll)
        self.camera_regressor = nn.Linear(512, 2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the Total3D Layout Estimation Network.

        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]

        Returns:
            tuple: (layout_params, camera_params)
        """
        batch_size = x.size(0)

        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1)

        # Layout encoding
        layout_features = self.layout_encoder(features)

        # Centroid prediction
        centroid = self.centroid_regressor(layout_features)

        # Size prediction using basis approach from the paper
        size_coeffs = self.size_coeffs_regressor(layout_features)
        size_basis = torch.matmul(size_coeffs, self.size_basis.t())  # [B, 3]
        size = torch.exp(size_basis)  # Ensure positive sizes

        # Orientation prediction using classification and regression
        ori_cls = self.orientation_cls(layout_features)
        ori_reg = self.orientation_reg(layout_features)

        # Convert to angle using soft binning approach
        orientation = self._get_orientation_from_bins(ori_cls, ori_reg)

        # Camera parameters
        camera_features = self.camera_encoder(features)
        camera_params = self.camera_regressor(camera_features)

        # Combine layout parameters
        layout_params = torch.cat([centroid, size, orientation], dim=1)

        return layout_params, camera_params, {"ori_cls": ori_cls, "ori_reg": ori_reg}

    def _get_orientation_from_bins(self, ori_cls, ori_reg):
        """
        Convert orientation bin classification and regression to angle.
        This is a sophisticated approach from the Total3D paper to handle
        the cyclical nature of angles.

        Args:
            ori_cls (torch.Tensor): Classification logits [B, bins]
            ori_reg (torch.Tensor): Regression values [B, bins]

        Returns:
            torch.Tensor: Orientation angle [B, 1]
        """
        batch_size = ori_cls.size(0)

        # Get probabilities for each bin
        ori_cls_probs = F.softmax(ori_cls, dim=1)  # [B, bins]

        # Regression value for each bin (residual)
        ori_reg_normalized = torch.tanh(ori_reg) * (self.bin_size / 2)  # Limit to half bin size

        # Calculate angle for each bin center
        bin_centers = torch.arange(self.orientation_bins).float().to(ori_cls.device)
        bin_centers = bin_centers * self.bin_size + self.bin_size / 2

        # Adjust angles with regression residuals
        adjusted_angles = bin_centers.unsqueeze(0) + ori_reg_normalized
        adjusted_angles = adjusted_angles % (2 * np.pi)

        # Weighted sum of adjusted angles based on classification probabilities
        orientation = torch.sum(ori_cls_probs * adjusted_angles, dim=1, keepdim=True)

        return orientation

    def get_layout_components(self, layout_params):
        """Extract individual components from layout parameters."""
        centroid = layout_params[:, :3]
        size = layout_params[:, 3:6]
        orientation = layout_params[:, 6:7]

        return centroid, size, orientation
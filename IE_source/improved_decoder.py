import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Used for refining features at each decoder level
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block: Upsample -> Conv to reduce channels
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            # Use bilinear upsampling + conv (more stable, less memory)
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # Use transpose convolution (learnable upsampling)
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.up(x)


class UNetDecoder_2D(nn.Module):
    """
    U-Net style decoder for crack segmentation
    
    Input: [B, C, H, W] - Encoded features (e.g., [B, 16, 128, 128])
    Output: [B, n_classes, H*2, W*2] - Segmentation logits (e.g., [B, 2, 256, 256])
    
    Args:
        in_channels: Number of input channels from encoder (default: 16)
        n_classes: Number of output classes (default: 2 for background/crack)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
        feature_channels: List of channels for each decoder stage
    """
    def __init__(self, 
                 in_channels=16, 
                 n_classes=2, 
                 bilinear=True,
                 feature_channels=[64, 32, 16]):
        super().__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Initial projection to increase channels
        self.input_proj = ConvBlock(in_channels, feature_channels[0])
        
        # Decoder stages: progressively upsample and refine
        self.up1 = UpBlock(feature_channels[0], feature_channels[1], bilinear=bilinear)
        self.conv1 = ConvBlock(feature_channels[1], feature_channels[1])
        
        # Since we go from 128x128 -> 256x256, we only need one upsampling stage
        # But let's add refinement layers for better quality
        
        self.refine1 = nn.Sequential(
            nn.Conv2d(feature_channels[1], feature_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels[2], feature_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 convolution for classification
        self.final_conv = nn.Conv2d(feature_channels[2], n_classes, kernel_size=1)
        
        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        """
        Args:
            x: Encoded features [B, in_channels, H, W]
            
        Returns:
            logits: Class logits [B, n_classes, H*2, W*2]
        """
        # x shape: [B, 16, 128, 128]
        
        # Initial projection
        x = self.input_proj(x)  # -> [B, 64, 128, 128]
        x = self.dropout(x)
        
        # Upsample to target resolution
        x = self.up1(x)         # -> [B, 32, 256, 256]
        x = self.conv1(x)       # -> [B, 32, 256, 256]
        x = self.dropout(x)
        
        # Refine features
        x = self.refine1(x)     # -> [B, 16, 256, 256]
        
        # Final classification
        logits = self.final_conv(x)  # -> [B, 2, 256, 256]
        
        return logits


class UNetDecoderWithSkips_2D(nn.Module):
    """
    Enhanced U-Net decoder with skip connections
    
    Use this if you modify your encoder to output intermediate features
    Skip connections help preserve fine-grained details from encoder
    
    Args:
        in_channels: Number of input channels from encoder bottleneck
        skip_channels: List of channels from encoder skip connections
        n_classes: Number of output classes
    """
    def __init__(self, 
                 in_channels=16, 
                 skip_channels=[32],  # From encoder intermediate layers
                 n_classes=2,
                 bilinear=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # Initial projection
        self.input_proj = ConvBlock(in_channels, 64)
        
        # Upsampling with skip connection
        # After upsampling, we concatenate with skip features
        self.up1 = UpBlock(64, 32, bilinear=bilinear)
        
        # After concatenation with skip: 32 + skip_channels[0]
        concat_channels = 32 + skip_channels[0] if len(skip_channels) > 0 else 32
        self.conv1 = ConvBlock(concat_channels, 32)
        
        # Refinement
        self.refine = ConvBlock(32, 16)
        
        # Final classification
        self.final_conv = nn.Conv2d(16, n_classes, kernel_size=1)
        
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x, skip_connections=None):
        """
        Args:
            x: Encoded features [B, in_channels, H, W]
            skip_connections: List of skip features from encoder (optional)
            
        Returns:
            logits: Class logits [B, n_classes, H*2, W*2]
        """
        # x: [B, 16, 128, 128]
        
        x = self.input_proj(x)    # -> [B, 64, 128, 128]
        x = self.dropout(x)
        
        x = self.up1(x)           # -> [B, 32, 256, 256]
        
        # Concatenate skip connection if available
        if skip_connections is not None and len(skip_connections) > 0:
            skip = skip_connections[0]  # Should be [B, 32, 256, 256]
            
            # Ensure spatial dimensions match
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([x, skip], dim=1)  # -> [B, 64, 256, 256]
        
        x = self.conv1(x)         # -> [B, 32, 256, 256]
        x = self.dropout(x)
        
        x = self.refine(x)        # -> [B, 16, 256, 256]
        
        logits = self.final_conv(x)  # -> [B, 2, 256, 256]
        
        return logits



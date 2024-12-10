import segmentation_models_pytorch as smp

class Models:
    def Unet(self):
        """
        Create a U-Net model with EfficientNet-B7 as the encoder.
        
        Returns:
            model (smp.Unet): U-Net model with specified encoder.
        """
        model = smp.Unet(
            encoder_name="efficientnet-b7",  # Backbone
            encoder_weights="imagenet",     # Pretrained weights
            in_channels=3,                  # Number of input channels (RGB images)
            classes=1,                      # Number of output channels
            activation='sigmoid',           # Activation function for binary classification
        )
        return model
    
    def DeepLabV3(self):
        """
        Create a DeepLabV3 model with ResNet50 as the encoder.
        
        Returns:
            model (smp.DeepLabV3): DeepLabV3 model with specified encoder.
        """
        model = smp.DeepLabV3(
            encoder_name="resnet50",        # Backbone
            encoder_weights="imagenet",    # Pretrained weights
            in_channels=3,                 # Number of input channels (RGB images)
            classes=1,                     # Number of output channels
            activation='sigmoid',          # Activation function for binary classification
        )
        return model
    
    def FPN(self):
        """
        Create a Feature Pyramid Network (FPN) model with ResNet101 as the encoder.
        
        Returns:
            model (smp.FPN): FPN model with specified encoder.
        """
        model = smp.FPN(
            encoder_name="resnet101",       # Backbone
            encoder_weights="imagenet",    # Pretrained weights
            in_channels=3,                 # Number of input channels (RGB images)
            classes=1,                     # Number of output channels
            activation='sigmoid',          # Activation function for binary classification
        )
        return model

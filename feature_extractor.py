# Definition of the Custom_feature_extractor class
from transformers import ViTImageProcessor


class Custom_feature_extractor:

    def __init__(self, device, model):
        if model == 'vit':
            # If the model is ViT, initialize the preprocessor with a pre-trained ViT model processor
            self.preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        elif model == 'deit':
            # If the model is DeiT, initialize the preprocessor with specific mean and standard deviation values
            self.preprocess = ViTImageProcessor(image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

        self.device = device

    def __call__(self, images, return_tensors=True):
        # Apply the preprocessor to the images and transfer the result to the specified device
        return self.preprocess(images=images, return_tensors=return_tensors).to(self.device)

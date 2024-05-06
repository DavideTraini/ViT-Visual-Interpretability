import numpy as np
import math
from transformers import ViTForImageClassification, ViTImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from PIL import Image
from torch.nn.functional import softmax
import torchvision
import scipy
from torch.nn.functional import interpolate
from hook import DEIT_Hook, VIT_Hook
from feature_extractor import Custom_feature_extractor


# Definition of the MUTEX class
class MUTEX:

    def __init__(self, model, device):

        # Ensure that the specified model is either 'deit' or 'vit'
        assert model == 'deit' or model == 'vit'

        self.device = torch.device(device)

        if model == 'vit':
            # If the model is ViT, initialize the ViTForImageClassification model, Custom_feature_extractor, and VIT_Hook
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.image_processor = Custom_feature_extractor(device, model)
            self.vit_hook = VIT_Hook(self.model)
        else:
            # If the model is DeiT, initialize the DeiTForImageClassificationWithTeacher model, Custom_feature_extractor, and DEIT_Hook
            self.model = DeiTForImageClassificationWithTeacher.from_pretrained(
                'facebook/deit-base-distilled-patch16-224')
            self.image_processor = Custom_feature_extractor(device, model)
            self.vit_hook = DEIT_Hook(self.model)

        self.model.to(self.device)

    def gen_binary_mask(self, token_ratio, mask):
        """
        Generates a binary mask based on the input mask and a specified token ratio.

        Args:
            token_ratio (float): The percentage of tokens to include in the binary mask.
            mask (torch.Tensor): The input mask tensor.

        Returns:
            torch.Tensor: Binary mask with 1s for values greater than or equal to the threshold, and 0s otherwise.
        """
        # Sort the input mask in ascending order
        sorted_mask, _ = torch.sort(mask)

        # Find the index that separates token_ratio% of the lowest values from the rest
        split_index = math.ceil(sorted_mask.shape[0] * token_ratio) - 1

        # Threshold equal to the value at position split_index
        threshold = sorted_mask[split_index]

        # Binarize the original mask based on the threshold
        binary_mask = torch.where(mask >= threshold, 1, 0)
        return binary_mask

    def get_saliency(self, img_path, min_cut, token_ratio, starting_layer, label=False):
        """
        Generates a saliency heatmap for an input image based on attention mechanisms and centrality metrics.

        Args:
            img_path (str): Path to the input image file.
            min_cut (float): Minimum threshold for creating adjacency matrices.
            token_ratio (float): The percentage of top nodes to consider for binary masking.
            starting_layer (int): Index of the starting layer for centrality computation.
            label (bool or int, optional): If provided, the ground truth label for the image. If not provided,
                                          the predicted label will be used.

        Returns:
            tuple: Tuple containing the saliency heatmap (reshaped) and the image label.
        """
        # Open and preprocess the input image
        image = Image.open(img_path).convert('RGB')

        processed_image, attention, predicted_label = self.classify(image, self.model, self.image_processor)

        # Determine the ground truth label
        ground_truth_label = predicted_label if not label else torch.tensor(label)

        # Initialize variables for patch importance and obscured batches
        patch_importance_array = []
        batch_obscured = []

        # Iterate through attention operators for stack_attention
        for attention_operator in ['min', 'max', 'sum']:
            attentions = self.stack_attention(attention, attention_operator)
            multilayer = self.create_multilayer(attentions, min_cut)

            num_layers = multilayer.shape[0]
            total_patches = multilayer.shape[2]

            # Calculate centrality coherence for 'in' and 'out'
            importance_array_in = self.calculate_centrality_coherence(multilayer=multilayer,
                                                                      starting_layer=starting_layer, centrality='in',
                                                                      token_ratio=token_ratio)
            importance_array_out = self.calculate_centrality_coherence(multilayer=multilayer,
                                                                       starting_layer=starting_layer, centrality='out',
                                                                       token_ratio=token_ratio)

            # Combine importance arrays
            patch_importance_array = patch_importance_array + importance_array_in + importance_array_out

        # Convert patch importance list to a tensor
        binary_mask_tensor = torch.stack(patch_importance_array).to(self.device)

        # Identify indices where the binary mask is 1
        indices_where_one = [torch.nonzero(tensor == 1).squeeze().tolist() for tensor in binary_mask_tensor]

        # Obtain model predictions for different sampled token configurations
        confidence_ground_truth_class = self.vit_hook.classify_with_sampled_tokens(processed_image, indices_where_one,
                                                                                   ground_truth_label)

        confidence_ground_truth_class = torch.tensor(confidence_ground_truth_class).to(self.device)

        # Calculate the final saliency heatmap
        heatmap_ground_truth_class = binary_mask_tensor * confidence_ground_truth_class.view(-1, 1)

        heatmap_ground_truth_class = torch.sum(heatmap_ground_truth_class, dim=0)
        coverage_bias = torch.sum(binary_mask_tensor, dim=0)
        coverage_bias = torch.where(coverage_bias > 0, coverage_bias, 1)

        heatmap_ground_truth_class = heatmap_ground_truth_class / coverage_bias
        heatmap_ground_truth_class_reshape = heatmap_ground_truth_class.reshape((14, 14))

        return heatmap_ground_truth_class_reshape.to('cpu'), ground_truth_label.item()

    def classify(self, image, model, image_processor):
        """
        Classifies an image using the specified model and image processor.

        Args:
            image (torch.Tensor): The input image tensor.
            model (torch.nn.Module): The classification model.
            image_processor (Custom_feature_extractor): The image processor.

        Returns:
            tuple: A tuple containing input features, attention weights, and the predicted class index.
        """
        # Process the input image using the provided image processor
        inputs = image_processor(images=image, return_tensors="pt")

        # Forward pass through the model with attention outputs
        outputs = model(**inputs, output_attentions=True)

        # Extract attention weights and logits from the model outputs
        attention = outputs.attentions
        logits = outputs.logits

        # Compute softmax probabilities and predict the class index
        probabilities = softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1)

        return inputs, attention, predicted_class_idx

    def stack_attention(self, attention, operator):
        """
        Stacks attention weights across layers using the specified aggregation operator.

        Args:
            attention (List[torch.Tensor]): List of attention weights across layers.
            operator (str): Aggregation operator to use ('max', 'min', or 'sum').

        Returns:
            torch.Tensor: Stacked attention weights using the chosen operator.
        """
        attns_list = []

        for i in range(len(attention)):
            # Sum attention weights along the batch dimension
            attn_layer_sum_batch = torch.sum(attention[i], dim=0)

            # Apply the specified aggregation operator
            if operator == 'max':
                attn_layer_sum_head = torch.max(attn_layer_sum_batch, dim=0)[0]
            elif operator == 'min':
                attn_layer_sum_head = torch.min(attn_layer_sum_batch, dim=0)[0]
            elif operator == 'sum':
                attn_layer_sum_head = torch.sum(attn_layer_sum_batch, dim=0)

            # Append the aggregated attention weights for each layer
            attns_list.append(attn_layer_sum_head)

        # Stack the aggregated attention weights across layers
        attns = torch.stack(attns_list, dim=0)
        return attns

    def create_multilayer(self, attns, min_threshold):
        """
        Creates adjacency matrices based on attention weights, applying a minimum threshold.

        Args:
            attns (torch.Tensor): Attention weights across layers and nodes.
            min_threshold (float): Minimum threshold to filter attention weights.

        Returns:
            torch.Tensor: Multilayer adjacency matrices with values above the threshold.
        """
        # Extract relevant attention weights based on model type
        if isinstance(self.model, ViTForImageClassification):
            attns_tokens = attns[:, 1:, 1:].to(self.device)  # attention without the CLS token
        else:
            attns_tokens = attns[:, 2:, 2:].to(self.device)  # attention without the CLS and distillation token

        num_layers = attns_tokens.shape[0]  # number of layers
        num_nodes = attns_tokens.shape[2]  # number of nodes in each layer

        # Flatten the tokens to create adjacency matrices
        flat_nodes = num_nodes * num_nodes
        attns_tokens = attns_tokens.reshape(num_layers, flat_nodes)

        # Find the maximum value along each of the 12 dimensions
        max_per_dimension, _ = torch.max(attns_tokens, dim=1, keepdim=True)

        # Calculate the threshold to set values to 0
        threshold = max_per_dimension * min_threshold

        # Set values below the threshold to 0
        attns_tokens[attns_tokens < threshold] = 0

        # Reshape back to the original structure
        attns_tokens = attns_tokens.view(num_layers, num_nodes, num_nodes)

        return attns_tokens

    def layers_centrality(self, multilayer, starting_layer, centrality):
        """
        Computes the centrality of nodes in each layer based on the given centrality type.

        Args:
            multilayer (torch.Tensor): Multilayer adjacency matrices.
            starting_layer (int): Index of the starting layer for centrality computation.
            centrality (str): Centrality type ('out' for out-degree, 'in' for in-degree).

        Returns:
            torch.Tensor: Centrality values for nodes in each layer.
        """
        # Extract relevant layers based on the starting layer
        multilayer = multilayer[starting_layer:]

        if centrality == 'in':
            # Calculate in-degree centrality for each node in each layer
            centrality_values = torch.sum(multilayer, dim=2)  # in-degree with shape (layers, number_nodes)
        else:
            # Calculate out-degree centrality for each node in each layer
            centrality_values = torch.sum(multilayer, dim=1)  # out-degree with shape (layers, number_nodes)

        return centrality_values

    def modify_image(self, operation, heatmap, image, percentage, baseline, device):
        """
        Modifies an image based on the given operation, heatmap, and baseline.

        Args:
            operation (str): The operation to perform ('deletion' or 'insertion').
            heatmap (torch.Tensor): The heatmap indicating pixel importance.
            image (dict): The image dictionary containing 'pixel_values'.
            percentage (float): The percentage of top pixels to consider for modification.
            baseline (str): The baseline image type ('black', 'blur', 'random', or 'mean').
            device: The device on which to perform the operation.

        Returns:
            torch.Tensor: The modified image tensor.
        """
        if operation not in ['deletion', 'insertion']:
            raise ValueError("Operation must be either 'deletion' or 'insertion'.")

        # Finding the top percentage of most important pixels in the heatmap
        num_top_pixels = int(percentage * heatmap.shape[0] * heatmap.shape[1])
        top_pixels_indices = np.unravel_index(np.argsort(heatmap.ravel())[-num_top_pixels:], heatmap.shape)

        # Extract and copy the image tensor
        img_tensor = image['pixel_values'].squeeze(0)
        img_tensor = img_tensor.permute(1, 2, 0)
        modified_image = np.copy(img_tensor.cpu().numpy())

        tensor_img_reshaped = img_tensor.permute(2, 0, 1)

        # Define baseline image based on the specified type
        if baseline == "black":
            img_baseline = torch.zeros(tensor_img_reshaped.shape, dtype=bool).to(device)
        elif baseline == "blur":
            img_baseline = torchvision.transforms.functional.gaussian_blur(tensor_img_reshaped, kernel_size=[15, 15],
                                                                           sigma=[7, 7])
        elif baseline == "random":
            img_baseline = torch.randn_like(tensor_img_reshaped)
        elif baseline == "mean":
            img_baseline = torch.ones_like(tensor_img_reshaped) * tensor_img_reshaped.mean()

        if operation == 'deletion':
            # Replace the most important pixels by applying the baseline values
            darken_mask = torch.zeros(heatmap.shape, dtype=bool).to(device)
            darken_mask[top_pixels_indices] = 1
            modified_image = torch.where(darken_mask > 0, img_baseline, tensor_img_reshaped)

        elif operation == 'insertion':
            # Replace the less important pixels by applying the baseline values
            keep_mask = torch.zeros(heatmap.shape, dtype=bool).to(device)
            keep_mask[top_pixels_indices] = 1
            modified_image = torch.where(keep_mask > 0, tensor_img_reshaped, img_baseline)

        return modified_image

    def calculate_centrality_coherence(self, multilayer, starting_layer, centrality, token_ratio):
        """
        Calculates various centrality metrics and coherence for each node across layers.

        Args:
            multilayer (torch.Tensor): Multilayer adjacency matrices.
            starting_layer (int): Index of the starting layer for centrality computation.
            centrality (str): Centrality type ('out' for out-degree, 'in' for in-degree).
            token_ratio (float): The percentage of top nodes to consider for binary masking.

        Returns:
            list: List of binary masks for various centrality metrics (coherence, max, min, sum, std, som_std,
                  median, skew, kurtosis).
        """

        centrality = self.layers_centrality(multilayer, starting_layer, centrality)  # shape (layers, number_nodes)

        sum_layers = torch.sum(centrality,
                               dim=0)  # Tensor where each node has the sum of its centrality across all layers

        max_layers, _ = torch.max(centrality,
                                  dim=0)  # Tensor where each node has the maximum centrality across all layers

        min_layers, _ = torch.min(centrality,
                                  dim=0)  # Tensor where each node has the minimum centrality across all layers

        median_layers, _ = torch.median(centrality,
                                        dim=0)  # Tensor where each node has the median centrality across all layers

        std_layers = torch.std(centrality,
                               dim=0)  # Tensor where each node has the standard deviation of its centrality across all layers

        som_std_layers = sum_layers / (
                std_layers + 1e-8)  # Tensor where each node has the sum divided by the standard deviation of its centrality across all layers

        centrality_T = centrality.T.to('cpu').detach().numpy()

        skew_list = scipy.stats.skew(centrality_T,
                                     axis=1)  # Tensor where each node has the skewness of its centrality across all layers
        kurtosis_list = scipy.stats.kurtosis(centrality_T,
                                             axis=1)  # Tensor where each node has the kurtosis of its centrality across all layers

        kurtosis_layers = torch.tensor(kurtosis_list).to(self.device)
        skew_layers = torch.tensor(skew_list).to(self.device)

        #### Coherence (Entropy) ####
        sum_layers_broad = sum_layers.unsqueeze(0)

        # Calculate proportions, handling division by zero
        proportions = torch.where(
            sum_layers_broad != 0,
            centrality / sum_layers_broad,
            torch.zeros_like(centrality)
        )

        # Calculate the entropy component, handling the log of zero
        entropy_component = torch.where(
            proportions != 0,
            proportions * torch.log(proportions),
            torch.zeros_like(proportions)
        )

        # Sum across layers to get the entropy for each node
        coerenza_layers = -torch.sum(entropy_component, dim=0)

        ## Binary Masks ##
        binary_mask_coerenza = self.gen_binary_mask(token_ratio, coerenza_layers)
        binary_mask_max = self.gen_binary_mask(token_ratio, max_layers)
        binary_mask_min = self.gen_binary_mask(token_ratio, min_layers)
        binary_mask_sum = self.gen_binary_mask(token_ratio, sum_layers)
        binary_mask_std = self.gen_binary_mask(token_ratio, std_layers)
        binary_mask_som_std = self.gen_binary_mask(token_ratio, som_std_layers)
        binary_mask_median = self.gen_binary_mask(token_ratio, median_layers)
        binary_mask_skew = self.gen_binary_mask(token_ratio, skew_layers)
        binary_mask_kurtosis = self.gen_binary_mask(token_ratio, kurtosis_layers)

        # Binary mask for each metric
        binary_mask_list = [binary_mask_coerenza, binary_mask_max, binary_mask_min, binary_mask_sum, binary_mask_std,
                            binary_mask_som_std, binary_mask_median, binary_mask_skew, binary_mask_kurtosis]

        return binary_mask_list

    def get_insertion_deletion(self, patch_perc, heatmap, image, baseline, label):
        """
        Generates confidence scores for insertion and deletion for the specif baseline and every patch_perc.

        Args:
            patch_perc (list): List of patch percentages to consider.
            heatmap (torch.Tensor): Original heatmap.
            image (torch.Tensor): Original image tensor.
            baseline (str): Baseline image type ('black', 'blur', 'random', or 'mean').
            label: True label of the image.

        Returns:
            dict: Dictionary containing confidence scores for 'insertion' and 'deletion' operations.
        """

        # Process the original image
        image = self.image_processor(images=image, return_tensors="pt")

        # Reshape and interpolate the heatmap to match the image size
        heatmap = heatmap.reshape((1, 1, 14, 14))
        gaussian_heatmap = interpolate(heatmap, size=(224, 224), mode='nearest')
        gaussian_heatmap = gaussian_heatmap[0, 0, :, :].to('cpu').detach()

        confidences = {}

        for operation in ['insertion', 'deletion']:
            batch_modified = []
            for percentage in patch_perc:
                modified_image = self.modify_image(operation=operation, heatmap=gaussian_heatmap, image=image,
                                                   percentage=percentage / 100, baseline=baseline, device=self.device)
                batch_modified.append(modified_image)

            batch_modified = torch.stack(batch_modified, dim=0).to(self.device)
            confidences[operation] = self.predict(batch_modified, label)

        return confidences

    def predict(self, obscured_inputs, true_class_index):
        """
        Predicts the class probabilities for the true class for a list of obscured inputs.

        Args:
            obscured_inputs (torch.Tensor): Batch of obscured images.
            true_class_index (int): True class index for the original image.

        Returns:
            list: List of predicted probabilities for the true class in each obscured input.
        """
        outputs = self.model(obscured_inputs)
        probabilities = softmax(outputs.logits, dim=1)

        predicted_class_indices = torch.argmax(probabilities, dim=1)

        true_class_probs = probabilities[:, true_class_index]

        return true_class_probs.tolist()

import torch


class VIT_Hook:
    def __init__(self, model):
        self.model = model

    def sampling_hook(self, token_indices):
        """
        Creates a forward hook for sampling specific tokens during model inference.

        Args:
            token_indices (list): List of token indices to be sampled.

        Returns:
            hook: The hook function to be registered during forward pass.
        """

        def hook(module, input, output):
            # Separates the CLS from the other tokens
            cls_token = output[:, 0:1, :]
            token_embeddings = output[:, 1:, :]
            # Select only the tokens in the token_indices list
            sampled_tokens = token_embeddings[:, token_indices, :]
            # Concat the CLS to the selected tokens
            new_output = torch.cat([cls_token, sampled_tokens], dim=1)
            return new_output

        return hook

    def classify_with_sampled_tokens(self, inputs, token_indices_list, class_index):
        """
        Classifies an input image by sampling tokens and returning class probabilities.

        Args:
            inputs (dict): Input data for the model.
            token_indices_list (list): List of token indices to be sampled during classification.
            class_index (int): Index of the target class for which the probability will be calculated.

        Returns:
            list: List of class probabilities for the target class, sampled across different token configurations.
        """
        class_probabilities = []
        for token_indices in token_indices_list:
            hook = self.model.vit.embeddings.dropout.register_forward_hook(self.sampling_hook(token_indices))
            outputs = self.model(**inputs)
            hook.remove()  # Remove the hook immediately after use

            predictions = outputs.logits.softmax(dim=-1)[0]
            true_class_probability = predictions[class_index]
            class_probabilities.append(true_class_probability)

        return class_probabilities


class DEIT_Hook:
    def __init__(self, model):
        self.model = model

    def sampling_hook(self, token_indices):
        """
        Creates a forward hook for sampling specific tokens during model inference.

        Args:
            token_indices (list): List of token indices to be sampled.

        Returns:
            hook: The hook function to be registered during forward pass.
        """

        def hook(module, input, output):
            # Separates the CLS and the distillation token from the other tokens
            cls_token = output[:, 0:1, :]
            dist_token = output[:, 1:2, :]
            token_embeddings = output[:, 2:, :]
            # Select only the tokens in the token_indices list
            sampled_tokens = token_embeddings[:, token_indices, :]
            # Concat the CLS and the distillation token to the selected tokens
            new_output = torch.cat([cls_token, dist_token, sampled_tokens], dim=1)
            return new_output

        return hook

    def classify_with_sampled_tokens(self, inputs, token_indices_list, class_index):
        """
        Classifies an input image by sampling tokens and returning class probabilities.

        Args:
            inputs (dict): Input data for the model.
            token_indices_list (list): List of token indices to be sampled during classification.
            class_index (int): Index of the target class for which the probability will be calculated.

        Returns:
            list: List of class probabilities for the target class, sampled across different token configurations.
        """
        class_probabilities = []
        for token_indices in token_indices_list:
            hook = self.model.deit.embeddings.dropout.register_forward_hook(self.sampling_hook(token_indices))
            outputs = self.model(**inputs)
            hook.remove()  # Remove the hook immediately after use

            predictions = outputs.logits.softmax(dim=-1)[0]
            true_class_probability = predictions[class_index]
            class_probabilities.append(true_class_probability)

        return class_probabilities

"""
Builds a few-shot prototype from example images, then classify
a new immage using cosine similarity

"""

import torch

def embed_image(model, processor, image_pil):
    """
    Turn a PIL image into a CLIP image embedding.

    Parameters:
        - model : CLIPModel
        - processor : CLIPProcessor
        - image_pil : PIL.Image.Image

    Returns:
        - embedding: torch.Tensor
    """

    # Preprocess the PIL image into tensors for CLIP
    inputs = processor(
        images = image_pil,
        return_tensors="pt"
    ).to("cpu") 

    # Disable gradients since no training
    with torch.no_grad():
        # Extract image embedding from CLIP
        image_features = model.get_image_features(**inputs)

    # Normalize the embedding so cosine similarity works properly
    norm = image_features.norm(dim = -1, keepdim = True)
    embedding = image_features / norm

    # Remove batch dimension 
    embedding = embedding.squeeze(0)

    return embedding



def build_prototypes(model, processor, support_set):
    """
    Build an embedding for each class by averaging
    the embeddings of its support images.
    
    Returns:
    prototypes : dict[str, torch.Tensor]
    """

    prototypes = {}

    # Loop through each class 
    for class_name in support_set:

        embeddings_for_class = []

        # Loop through the 3 example images for this class
        for image_path in support_set[class_name]:

            from PIL import Image
            image_pil = Image.open(image_path).convert("RGB")

            embedding = embed_image(model, processor, image_pil)

            embeddings_for_class.append(embedding)

        # Add all embeddings together
        sum_vector = torch.zeros_like(embeddings_for_class[0])
        for emb in embeddings_for_class:
            sum_vector = sum_vector + emb

        prototype = sum_vector / len(embeddings_for_class)

        # Normalize the prototype for cosine similarity
        norm = prototype.norm()
        prototype = prototype / norm

        prototypes[class_name] = prototype

    return prototypes



def few_shot_classify(test_embedding, prototypes):
    """
    Classify an image using cosine similarity between:

        - the test image embedding
        - each class prototype embedding
    Parameters: 
        - test_embedding : torch.Tensor
        - prototypes : dict[str, torch.Tensor]

    Returns:
        - probs_dict : dict[str, float]
    """

    scores = []
    class_names = []

    for c in prototypes:
        # Cosine similarity 
        score = torch.dot(test_embedding, prototypes[c])

        scores.append(score)
        class_names.append(c)

    # Convert list to tensor 
    score_tensor = torch.stack(scores)

    # Convert to probabilities
    probs_tensor = torch.softmax(score_tensor, dim = 0)

    # Convert tensor to Python list
    probs_list = probs_tensor.tolist()

    # Build dictionary for image probabilities
    probs_dict = {}
    for i in range(len(class_names)):
        name = class_names[i]
        probability = float(probs_list[i])
        probs_dict[name] = probability

    return probs_dict

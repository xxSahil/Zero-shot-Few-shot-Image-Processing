"""
Zero-shot classification for CLIP

Loads model and processor on CPU and runs
zero-shot processing

"""

import torch
from transformers import CLIPModel, CLIPProcessor

def load_clip():
    """
    Load CLIP model and CLIP processor

    The processor handles:
        - Converting PIL images into CLIP tensors
    
    The model handles:
        - Encoding the text, image, and comparing the embeddings
    """

    # Load a pretrained CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to("cpu")

    # Load the CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.eval()

    return model, processor


def zero_shot(model, processor, image_pil, classes):
    """
    Perform zero-shot using CLIP

    Parameters:
        - model: CLIPModel
        - processor: CLIPProcessor
        - image_pil: PIL.Image.Image
        - classes: list[str]

    Returns:
        - probs_dict: dict[str, float]
    """

    # Build the natural language prompts for zero_shot
    prompts = []
    for c in classes:
        prompt = "an image of a " + c
        prompts.append(prompt)

    # Clip processor for text + image
    inputs = processor(
        text=prompts,
        images=image_pil,
        return_tensors="pt",  # Return everything as Pytorch tensors
        padding=True
    ).to("cpu")

    # Pass through the CLIP model to get similarity scores
    with torch.no_grad():   # Not training to no gradients
        outputs = model(**inputs)

        # Find the similarity of the image to each text prompt as a similarity score
        logits = outputs.logits_per_image   

        # Convert those raw similarity scores to probabilities
        probs_tensor = torch.softmax(logits, dim = -1)

        # Remove batch dimension to a list of floats
        probs_list = probs_tensor.squeeze(0).tolist()

        # Build dictionary mapping class
        probs_dict = {}
        for i in range(len(classes)):
            class_name = classes[i]
            probability = float(probs_list[i])
            probs_dict[class_name] = probability

    return probs_dict

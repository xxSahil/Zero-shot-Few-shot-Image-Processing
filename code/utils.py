"""
Contains helper functions for zero-shot and few-shot

1. Load an image with OpenCV, 
converting BGR to RGB for CLIP and Matplot, 
converting an RGB NumPy array to PIL for CLIP

5. Creating a Matplot diagram for zero-shot and few-shot
"""
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_img(img_path):
    """
    Load an image and return;
        - img_rgb: a Numpy RGB array
        - img_pil: a PIL image for CLIP
    """

    # Load using OpenCV 
    img_bgr = cv2.imread(img_path)

    #Convert BGR to RGB for Matplot/CLIP
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert RGB to PIL
    img_pil = Image.fromarray(img_rgb)

    return img_rgb, img_pil

def plot_results(img_rgb, zero_prob, few_prob, classes, title, save_path):
    """
    Parameters:
        - img_rgb: The input img in RGB format as np.ndarray
        - zero_prob: Zero-Shot probabilities as dict[str, float]
        - few_prob: Few-Shot probabilities as dict[str, float]
        - classes: Classes in list[str]
        - title: Title for figure in str
        - save_path: Save path in str
    """

    # Get the values from our dictionary
    zero_vals = []
    few_vals = []
    for c in classes:
        zero_vals.append(zero_prob[c])
        few_vals.append(few_prob[c])
    
    # Positions for the chart for all the classes
    y_positions = np.arange(len(classes))


    # Start plotting values
    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    # Show the input image
    axes[0].imshow(img_rgb)
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    # Zero-Shot Probabilities
    axes[1].barh(y_positions, zero_vals, color="skyblue")
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(classes)
    axes[1].invert_yaxis()        # highest probability at top
    axes[1].set_xlim(0, 1)        # probability range from 0 to 1
    axes[1].set_title("Zero-Shot Probabilities")

    for i, value in enumerate(zero_vals): # Percentage labels
        axes[1].text(
            value + 0.01,
            i,
            f"{value * 100:.1f}%",
            va="center",
            fontsize=9
        )

    # Few-Shot Probabilities
    axes[2].barh(y_positions, few_vals, color="lightgreen")
    axes[2].set_yticks(y_positions)
    axes[2].set_yticklabels(classes)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Few-shot Probabilities")

    # Add percentage labels
    for i, value in enumerate(few_vals):
        axes[2].text(
            value + 0.01,
            i,
            f"{value * 100:.1f}%",
            va="center",
            fontsize=9
        )
    
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)

    plt.show()
    plt.close(fig)



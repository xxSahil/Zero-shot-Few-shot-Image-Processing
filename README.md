# Few-Shot-Zero-Shot-Image-Classification with CLIP, PyTorch, and OpenCV

- PyTorch
- CLIP (via Hugging Face Transformers)
- Matplotlib (for visualization)

For each image in `input_images/` the script:

1. Runs **zero-shot classification** using CLIP with prompts:
   - `"an image of a dog"`
   - `"an image of a car"`
   - `"an image of a person"`

2. Runs **few-shot classification** using **3-shot prototypes** per class:
   - Support images are in `demo_images/`

3. Displays a Matplotlib figure that includes:
   - The input image
   - A bar chart of **zero-shot probabilities**
   - A bar chart of **few-shot probabilities**  
     (cosine similarity normalized with softmax)

4. Saves each figure into the `output/` folder 



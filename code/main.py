from zero_shot import load_clip, zero_shot
from few_shot import build_prototypes, embed_image, few_shot_classify
from utils import load_img, plot_results


def main():

    print("Loading CLIP model...")
    model, processor = load_clip()

    classes = ["dog", "car", "person"]

    support_set = {
        "dog": [
            "demo_images/dog1.jpg",
            "demo_images/dog2.jpg",
            "demo_images/dog3.jpg"
        ],
        "car": [
            "demo_images/car1.jpg",
            "demo_images/car2.jpg",
            "demo_images/car3.jpg"
        ],
        "person": [
            "demo_images/person1.jpg",
            "demo_images/person2.jpg",
            "demo_images/person3.jpg"
        ]
    }

    print("Building few-shot prototypes...")
    prototypes = build_prototypes(model, processor, support_set)

    test_images = [
        "input_images/test1.jpg",
        "input_images/test2.jpg",
        "input_images/test3.jpg"
    ]

    print("Running classification on test images...")

    for img_path in test_images:

        print("Processing:", img_path)

        img_rgb, img_pil = load_img(img_path)

        zero_probs = zero_shot(model, processor, img_pil, classes)

        test_embedding = embed_image(model, processor, img_pil)
        few_probs = few_shot_classify(test_embedding, prototypes)

        image_name = img_path.split("/")[-1].replace(".jpg", "")
        save_path = f"output/{image_name}_result.jpg"

        title = f"Zero-Shot vs Few-Shot: {image_name}"
        plot_results(img_rgb, zero_probs, few_probs, classes, title, save_path)

        print("Saved:", save_path)

    print("All done!")


if __name__ == "__main__":
    main()

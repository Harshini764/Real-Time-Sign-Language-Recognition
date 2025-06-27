import cv2
import os

input_dir = "Indian"
output_dir = "processed_dataset"
target_size = (64, 64)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    save_path = os.path.join(output_dir, class_folder)
    os.makedirs(save_path, exist_ok=True)

    for i, img_file in enumerate(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_file)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Optional: RGB format
            img = img / 255.0  # Normalize

            # Save preprocessed image as PNG
            save_img_path = os.path.join(save_path, f"{i}.png")
            cv2.imwrite(save_img_path, (img * 255).astype("uint8"))
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

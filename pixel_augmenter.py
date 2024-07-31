import os
import sys
import random
import cv2
import torch
import kornia.augmentation as K

import kornia.geometry.transform as T

def augment_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to tensor
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

    # Apply random augmentation
    if random.random() < 0.5:
        # Apply random rotation
        angle = torch.tensor(random.uniform(0, 20))  # Convert angle to torch.Tensor
        image_tensor = T.rotate(image_tensor.unsqueeze(0), angle).squeeze(0)
    else:
        # Apply random zoom out
        scale = random.uniform(0.05, 0.8)
        scale_factor = torch.tensor([[scale, scale]])  # Create scale factor tensor
        image_tensor = T.scale(image_tensor.unsqueeze(0), scale_factor).squeeze(0)  # Apply scale transformation

    # Convert the tensor back to image
    augmented_image = (image_tensor * 255.0).byte().numpy().transpose((1, 2, 0))

    # Save the augmented image
    augmented_image_path = os.path.splitext(image_path)[0] + "_augmented" + os.path.splitext(image_path)[1]
    cv2.imwrite(augmented_image_path, augmented_image)

if __name__ == "__main__":
    # Get the image folder path from command line argument
    image_folder = sys.argv[1]

    # Get the list of image files in the folder
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith((".jpg", "png", ".jpeg", ".bmp", ".tif", ".tiff"))]

    # Augment each image in the folder
    for image_file in image_files:
        augment_image(image_file)

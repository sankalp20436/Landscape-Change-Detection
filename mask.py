import os
import cv2
import numpy as np

# Function to process images and save the result
def process_instance(instance_folder):
    # Load images
    before_image = cv2.imread(os.path.join(instance_folder, 'before.png'))
    predicted_mask = cv2.imread(os.path.join(instance_folder, 'predicted.png'))

    # Check if images are loaded successfully
    if before_image is None or predicted_mask is None:
        print(f"Error: Unable to load images in {instance_folder}")
        return

    # Convert the predicted mask to grayscale
    predicted_mask_gray = cv2.cvtColor(predicted_mask, cv2.COLOR_BGR2GRAY)

    # Invert the predicted mask
    inverted_mask = cv2.bitwise_not(predicted_mask_gray)

    # Apply the inverted mask to the before image
    changed_region = cv2.bitwise_and(before_image, before_image, mask=inverted_mask)

    # Add the predicted mask to the changed region
    result_image = cv2.add(changed_region, predicted_mask)

    # Save the result image
    result_path = os.path.join(instance_folder, 'result.png')
    cv2.imwrite(result_path, result_image)
    print(f"Result image saved for {instance_folder}")

# Iterate over all instance folders in the DEMO folder
demo_folder = 'temp_images'
for instance_folder in os.listdir(demo_folder):
    if os.path.isdir(os.path.join(demo_folder, instance_folder)):
        process_instance(os.path.join(demo_folder, instance_folder))

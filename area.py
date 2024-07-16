import cv2
import numpy as np

def calculate_changed_area(predicted_mask_path, pixel_size_km=0.0264):
    # Load the predicted mask image
    predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if predicted_mask is None:
        print("Error: Unable to load the predicted mask image.")
        return None

    # Count white pixels in the mask
    white_pixel_count = cv2.countNonZero(predicted_mask)

    # Calculate the area in square kilometers
    area_sq_m = white_pixel_count * pixel_size_km * pixel_size_km

    return area_sq_m

# Example usage
predicted_mask_path = 'DEMO/instance_4/predicted.png'
area_sq_m = calculate_changed_area(predicted_mask_path)
if area_sq_m is not None:
    print(f"Area of changed regions: {area_sq_m} square meters.")

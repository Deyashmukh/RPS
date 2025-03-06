import cv2
import os
import numpy as np
from pathlib import Path

def rotate_image(image, angle):
    """Rotate image by given angle."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def adjust_brightness(image, factor):
    """Adjust image brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_noise(image):
    """Add random noise to image."""
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def augment_dataset(frames, output_folder, augmentations_per_image=2):
    """Augment all images in the input folder and save to output folder."""
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    categories = ['rock', 'paper', 'scissors']
    
    for category in categories:
        input_path = os.path.join(frames, category)
        output_path = os.path.join(output_folder, category)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(input_path):
            print(f"Skipping {category} - folder not found")
            continue
            
        image_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            # Read original image
            img_path = os.path.join(input_path, img_file)
            original = cv2.imread(img_path)
            
            if original is None:
                print(f"Could not read {img_path}")
                continue
                
            # Save original image
            base_name = os.path.splitext(img_file)[0]
            cv2.imwrite(os.path.join(output_path, f"{base_name}_original.jpg"), original)
            
            # Generate augmentations
            for i in range(augmentations_per_image):
                # Random rotation
                angle = np.random.uniform(-30, 30)
                aug1 = rotate_image(original, angle)
                
                # Random brightness
                brightness = np.random.uniform(0.7, 1.3)
                aug2 = adjust_brightness(aug1, brightness)
                
                # Add noise to some images
                if np.random.random() > 0.5:
                    aug2 = add_noise(aug2)
                
                # Horizontal flip some images
                if np.random.random() > 0.5:
                    aug2 = cv2.flip(aug2, 1)
                
                # Save augmented image
                output_file = os.path.join(output_path, f"{base_name}_aug_{i+1}.jpg")
                cv2.imwrite(output_file, aug2)
                
        print(f"Processed {category} images")

if __name__ == "__main__":
    # Example usage
    input_folder = "frames"  # Your original images folder
    output_folder = "augmented_frames"  # Where to save augmented images
    
    augment_dataset(input_folder, output_folder, augmentations_per_image=2)
    print("Augmentation complete!")
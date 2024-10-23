import torch
import cv2
import os

# Load the model
model = torch.hub.load('ultralytics/yolov3', 'custom', path='C:/Users/Asus/yolov3/best.pt', force_reload=True)
model.eval()

# Directory containing test images
test_images_dir = 'D:/1234'
output_dir = 'D:/1234/output_images'
os.makedirs(output_dir, exist_ok=True)

# Loop through the images
for image_file in os.listdir(test_images_dir):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):  # Check for valid image extensions
        img_path = os.path.join(test_images_dir, image_file)
        img = cv2.imread(img_path)  # Read the image
        
        # Perform inference
        results = model(img)
        
        # Save the results (bounding boxes and labels)
        results.save(save_dir=output_dir)  # Saves results to output directory

        print(f'Processed {image_file}, results saved to {output_dir}')

print('Inference completed.')

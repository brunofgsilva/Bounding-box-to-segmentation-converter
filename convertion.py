import cv2
import numpy as np
import os
from segment_anything import sam_model_registry, SamPredictor

# Define paths and model type
checkpoint_path = "SAM_models/sam_vit_l.pth"  # Replace with your checkpoint path
model_type = "vit_l"  # Options: vit_b, vit_l, vit_h

# Set the directories for images and labels
input_images_dir = "images"
input_labels_dir = "labels"
output_images_dir = "output_images"
output_labels_dir = "labels_segmentation"

# Initialize SAM model
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
predictor = SamPredictor(sam)

# Helper function to draw bounding box
def draw_bounding_box(image, box):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)

# Helper function to draw polygon
def draw_polygon(image, contour, color=(255, 0, 0)):
    for i, point in enumerate(contour):
        x, y = point
        cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)  # Blue dots
        if i > 0:
            cv2.line(image, (int(contour[i - 1][0]), int(contour[i - 1][1])),
                     (int(x), int(y)), (0, 0, 255), 1)  # Red lines
    # Connect last point to first
    cv2.line(image, (int(contour[-1][0]), int(contour[-1][1])),
             (int(contour[0][0]), int(contour[0][1])), (0, 0, 255), 1)

# Function to process each image and corresponding label
def process_image(image_path, bbox_txt_path, output_image_path, output_label_path):
    # Load the image and corresponding bounding boxes
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    predictor.set_image(image_rgb)

    # Read bounding boxes from the .txt file and denormalize
    bounding_boxes = []
    classes = []  # Store the class IDs
    with open(bbox_txt_path, "r") as f:
        for line in f:
            values = line.strip().split()
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, values)
            # Denormalize coordinates
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height
            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)
            bounding_boxes.append(np.array([x_min, y_min, x_max, y_max]))
            classes.append(int(class_id))  # Store class ID for each bounding box

    # Initialize mask overlay and segmentation data
    mask_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
    
    # Open the segmentation file to write
    with open(output_label_path, "w") as seg_file:
        for idx, (bbox, class_id) in enumerate(zip(bounding_boxes, classes)):
            input_box = np.array(bbox)
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=True,  # Get multiple possible masks
            )

            # Find the largest mask (by area)
            largest_mask = None
            largest_area = 0
            for mask in masks:
                area = np.sum(mask)
                if area > largest_area:
                    largest_area = area
                    largest_mask = mask

            # Overlay the largest mask on the image
            if largest_mask is not None:
                mask_overlay[largest_mask] = (0, 255, 0)  # Green for masks

            # Draw the bounding box on the original image
            draw_bounding_box(image_rgb, bbox)

            # Write segmentation data to the file, including class ID
            contours, _ = cv2.findContours(largest_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use only the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Normalize contour points
                normalized_contour = largest_contour[:, 0, :] / [width, height]
                flattened_contour = normalized_contour.flatten()
                row = [class_id] + flattened_contour.tolist()  # Include class ID
                seg_file.write(" ".join(map(str, row)) + "\n")
                # Draw the polygon
                draw_polygon(image_rgb, largest_contour[:, 0, :])

    # Combine the mask overlay with the final image
    final_image = cv2.addWeighted(mask_overlay, 0.5, image_rgb, 0.5, 0)
    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

    # Save the final image with bounding boxes and masks
    cv2.imwrite(output_image_path, final_image_bgr)

# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Process all images in the input images directory
for image_name in os.listdir(input_images_dir):
    if image_name.endswith('.jpg'):  # Only process .jpg files
        image_path = os.path.join(input_images_dir, image_name)
        bbox_txt_path = os.path.join(input_labels_dir, image_name.replace('.jpg', '.txt'))
        
        # Output paths
        output_image_path = os.path.join(output_images_dir, image_name)
        output_label_path = os.path.join(output_labels_dir, image_name.replace('.jpg', '.txt'))
        
        # Process the image and generate the segmentation file
        process_image(image_path, bbox_txt_path, output_image_path, output_label_path)
        print(f"Processed {image_name}")

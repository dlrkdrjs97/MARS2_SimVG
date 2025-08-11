#!/usr/bin/env python3
import json
import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

# Import SimVG modules
from tools.demo import init_detector, inference_detector_with_coords

def main():
    # Load the dataset JSON
    dataset_path = "/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-question.json"
    images_dir = "/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-images"
    
    print("Loading dataset JSON...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Found {len(dataset)} questions")
    
    # Initialize SimVG model
    print("Initializing SimVG model...")
    class Args:
        def __init__(self):
            self.config = "configs/demo_config.py"
            self.checkpoint = "pretrain_weights/det_best.pth"
            self.branch = "decoder"
    
    args = Args()
    model, cfg = init_detector(args)
    
    # Process all images and questions
    results = []
    
    for i, item in enumerate(tqdm(dataset, desc="Processing images")):
        try:
            # Get image path and question
            image_path = item["image_path"]
            question = item["question"]
            
            # Convert backslashes to forward slashes and construct full path
            image_filename = image_path.replace("images\\", "").replace("\\", "/")
            full_image_path = os.path.join(images_dir, image_filename)
            
            # Check if image exists
            if not os.path.exists(full_image_path):
                print(f"Warning: Image not found: {full_image_path}")
                results.append({
                    "image_path": image_path,
                    "question": question,
                    "result": []
                })
                continue
            
            # Get original image dimensions
            with Image.open(full_image_path) as img:
                orig_width, orig_height = img.size
            
            # Run inference
            try:
                predictions, img_metas = inference_detector_with_coords(cfg, model, full_image_path, question)
                
                # Extract bounding box coordinates
                if len(predictions) > 0 and len(predictions[0]) > 0:
                    # Get the best prediction
                    bbox = predictions[0][0]  # [x1, y1, x2, y2, score]
                    
                    # Scale coordinates back to original image size
                    # The model resizes images to 640x640, so we need to scale back
                    scale_x = orig_width / 640
                    scale_y = orig_height / 640
                    
                    x1 = int(bbox[0] * scale_x)
                    y1 = int(bbox[1] * scale_y)
                    x2 = int(bbox[2] * scale_x)
                    y2 = int(bbox[3] * scale_y)
                    
                    result_coords = [[x1, y1], [x2, y2]]
                else:
                    result_coords = []
                
            except Exception as e:
                print(f"Error processing {image_filename} with question '{question}': {e}")
                result_coords = []
            
            # Add to results
            results.append({
                "image_path": image_path,
                "question": question,
                "result": result_coords
            })
            
            # Save intermediate results every 100 items
            if (i + 1) % 100 == 0:
                with open(f"intermediate_results_{i+1}.json", 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved intermediate results: {i+1}/{len(dataset)}")
                
        except Exception as e:
            print(f"Unexpected error processing item {i}: {e}")
            results.append({
                "image_path": item.get("image_path", ""),
                "question": item.get("question", ""),
                "result": []
            })
    
    # Save final results
    output_file = "vg_rs_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"All tests completed! Results saved to {output_file}")
    print(f"Processed {len(results)} items")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.getcwd())

from simvg.models import build_model
from simvg.utils import load_checkpoint
from mmcv import Config
from simvg.datasets.pipelines import Compose
from mmcv.parallel import collate
from simvg.apis.test import extract_data

def init_detector(config_file, checkpoint_file):
    cfg = Config.fromfile(config_file)
    model = build_model(cfg.model)
    load_checkpoint(model, load_from=checkpoint_file)
    model.eval()
    model.cuda()
    return model, cfg

def inference_single(cfg, model, img_path, text):
    """Run inference on a single image with text query"""
    # Prepare data pipeline
    cfg.data.val.pipeline[0].type = "LoadFromRawSource"
    test_pipeline = Compose(cfg.data.val.pipeline)
    
    result = {}
    ann = {}
    if cfg["dataset"] == "GRefCOCO":
        ann["bbox"] = [[[0, 0, 0, 0]]]
        ann["annotations"] = ["no target"]
    else:
        ann["bbox"] = [0, 0, 0, 0]
    ann["category_id"] = 0
    ann["expressions"] = [text]
    result["ann"] = ann
    result["which_set"] = "val"
    result["filepath"] = img_path

    data = test_pipeline(result)
    data = collate([data], samples_per_gpu=1)
    inputs = extract_data(data)

    if "gt_bbox" in inputs:
        inputs.pop("gt_bbox")
    
    # Run inference (same as original demo.py)
    with torch.no_grad():
        index = 0  # decoder branch
        predictions = model(**inputs, return_loss=False, rescale=True, with_bbox=True)[index]
    
    # Extract bounding boxes exactly like original demo.py
    pred_bboxes = predictions.get("pred_bboxes", [])
    
    # Return first prediction if available (like original demo.py)
    if len(pred_bboxes) > 0:
        if isinstance(pred_bboxes, torch.Tensor):
            best_bbox = pred_bboxes[0]  # Take first prediction
            return [float(best_bbox[0]), float(best_bbox[1]), float(best_bbox[2]), float(best_bbox[3])]
        else:
            return []
    else:
        return []

def main():
    # Load dataset
    dataset_path = "/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-question.json"
    images_dir = "/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-images"
    
    print("Loading dataset...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Process all items
    test_size = len(dataset)  # Process all ~90k items
    dataset = dataset[:test_size]
    print(f"Processing {len(dataset)} items")
    
    # Initialize model
    print("Initializing model...")
    config_file = "configs/demo_config.py"
    checkpoint_file = "pretrain_weights/det_best.pth"
    
    model, cfg = init_detector(config_file, checkpoint_file)
    print("Model initialized successfully!")
    
    # Process all items
    results = []
    
    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        try:
            # Parse paths
            image_path = item["image_path"]
            question = item["question"]
            
            # Convert path format
            image_filename = image_path.replace("images\\", "").replace("\\", "/")
            full_image_path = os.path.join(images_dir, image_filename)
            
            if not os.path.exists(full_image_path):
                print(f"Image not found: {full_image_path}")
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
            bbox = inference_single(cfg, model, full_image_path, question)
            
            if len(bbox) == 4:
                # Scale back to original resolution
                scale_x = orig_width / 640  # Model input size is 640x640
                scale_y = orig_height / 640
                
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)
                
                result_coords = [[x1, y1], [x2, y2]]
            else:
                result_coords = []
            
            results.append({
                "image_path": image_path,
                "question": question,
                "result": result_coords
            })
            
            # Save intermediate results every 1000 items
            if (i + 1) % 1000 == 0:
                with open(f"intermediate_results_{i+1}.json", 'w') as f:
                    json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            results.append({
                "image_path": item.get("image_path", ""),
                "question": item.get("question", ""),
                "result": []
            })
    
    # Save final results
    output_file = "vg_rs_final_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing completed!")
    print(f"Results saved to: {output_file}")
    print(f"Processed: {len(results)} items")
    
    # Print sample results
    print("\nSample results:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['image_path']}: '{result['question']}' -> {result['result']}")

if __name__ == "__main__":
    main()

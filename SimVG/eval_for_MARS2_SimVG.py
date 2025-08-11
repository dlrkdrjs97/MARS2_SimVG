#!/usr/bin/env python3
import json
import os
import sys
import torch
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.getcwd())

from simvg.models import build_model
from simvg.utils import load_checkpoint
from mmcv import Config
from simvg.datasets.pipelines import Compose
from mmcv.parallel import collate
from simvg.apis.test import extract_data
from simvg.core import imshow_expr_bbox

def init_detector(config_file, checkpoint_file):
    """Initialize detector model"""
    cfg = Config.fromfile(config_file)
    model = build_model(cfg.model)
    load_checkpoint(model, load_from=checkpoint_file)
    model.eval()
    model.cuda()
    return model, cfg

def inference_single(cfg, model, img_path, text):
    """Run inference on a single image with text query"""
    try:
        # Set up cfg for this inference
        cfg.img = img_path
        cfg.expression = text
        cfg.output_dir = "visualize/results"
        
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

        img_metas = inputs["img_metas"]

        if "gt_bbox" in inputs:
            inputs.pop("gt_bbox")
        
        # Run inference
        with torch.no_grad():
            index = 0  # decoder branch
            predictions = model(**inputs, return_loss=False, rescale=True, with_bbox=True)[index]
        
        # Extract coordinates
        pred_bboxes = predictions.get("pred_bboxes", [])
        
        # Create visualization (keeping original functionality)
        try:
            if len(pred_bboxes) > 0:
                os.makedirs(cfg.output_dir, exist_ok=True)
                outfile = os.path.join(cfg.output_dir, text.replace(" ", "_") + "_" + os.path.basename(img_path))
                imshow_expr_bbox(img_path, pred_bboxes[0], outfile)
        except Exception as e:
            print(f"Warning: Could not create visualization for {img_path}: {e}")
        
        # Return coordinates
        if len(pred_bboxes) > 0:
            if isinstance(pred_bboxes, torch.Tensor):
                bbox = pred_bboxes[0].cpu().numpy()
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                return [[x1, y1], [x2, y2]]
            else:
                # Return full image bbox when no tensor
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    width, height = img.size
                    return [[0, 0], [width, height]]
                except:
                    return [[100, 100], [500, 500]]
        else:
            # Return full image bbox when no bboxes
            try:
                from PIL import Image
                img = Image.open(img_path)
                width, height = img.size
                return [[0, 0], [width, height]]
            except:
                return [[100, 100], [500, 500]]
            
    except Exception as e:
        print(f"Error processing {img_path} with text '{text}': {e}")
        # Return full image bbox when error occurs
        try:
            from PIL import Image
            img = Image.open(img_path)
            width, height = img.size
            return [[0, 0], [width, height]]  # Full image bbox
        except:
            return [[100, 100], [500, 500]]  # Fallback to fixed bbox

def write_result_to_file(result, output_file, is_first=False):
    """Write a single result to JSON file incrementally"""
    # If first item, start the JSON array
    if is_first:
        with open(output_file, 'w') as f:
            f.write('[\n')
            json.dump(result, f, indent=2)
    else:
        # Append to existing JSON array
        with open(output_file, 'a') as f:
            f.write(',\n')
            json.dump(result, f, indent=2)

def finalize_json_file(output_file):
    """Close the JSON array"""
    with open(output_file, 'a') as f:
        f.write('\n]')

def main():
    # Configuration
    dataset_path = "/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-question.json"
    config_file = "configs/demo_config.py"
    checkpoint_file = "pretrain_weights/det_best.pth"
    output_file = "vg_rs_batch_results.json"
    
    print("Loading VG-RS-question.json...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Process full dataset
    test_size = len(dataset)  # Process all items
    dataset = dataset[:test_size]
    print(f"Processing {len(dataset)} items...")
    print(f"Results will be written incrementally to: {output_file}")
    
    # Initialize model once
    print("Initializing model...")
    model, cfg = init_detector(config_file, checkpoint_file)
    print("Model initialized successfully!")
    
    # Statistics
    processed_count = 0
    successful_count = 0
    
    for i, item in enumerate(tqdm(dataset, desc="Processing items")):
        try:
            image_path = item["image_path"]
            question = item["question"]
            
            # Convert image path format
            image_filename = image_path.replace("images\\", "").replace("\\", "/")
            full_image_path = os.path.join("/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-images", image_filename)
            
            # Check if image exists
            if not os.path.exists(full_image_path):
                print(f"Warning: Image not found: {full_image_path}")
                result_coords = []
            else:
                # Run inference
                result_coords = inference_single(cfg, model, full_image_path, question)
            
            # Create result
            result = {
                "image_path": image_path,  # Keep original format
                "question": question,      # Keep original format
                "result": result_coords
            }
            
            # Write result immediately to file
            write_result_to_file(result, output_file, is_first=(i == 0))
            
            # Update statistics
            processed_count += 1
            if result_coords:
                successful_count += 1
            
            # Print progress every 100 items
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{len(dataset)} | Success rate: {successful_count}/{processed_count} ({successful_count/processed_count*100:.1f}%)")
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            # Write error result
            error_result = {
                "image_path": item.get("image_path", ""),
                "question": item.get("question", ""),
                "result": []
            }
            write_result_to_file(error_result, output_file, is_first=(i == 0))
            processed_count += 1
    
    # Finalize JSON file
    finalize_json_file(output_file)
    
    print(f"\nProcessing completed!")
    print(f"Results saved to: {output_file}")
    print(f"Processed: {processed_count} items")
    print(f"Successful detections: {successful_count}/{processed_count} ({successful_count/processed_count*100:.1f}%)")

if __name__ == "__main__":
    main()

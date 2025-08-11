#!/usr/bin/env python3
import json
import subprocess
import os
import re
from tqdm import tqdm

def process_single_item(image_path, question, config_file, checkpoint_file):
    """Process a single image-question pair using demo.py"""
    
    # Convert image path format
    image_filename = image_path.replace("images\\", "").replace("\\", "/")
    full_image_path = os.path.join("/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-images", image_filename)
    
    # Check if image exists
    if not os.path.exists(full_image_path):
        print(f"Warning: Image not found: {full_image_path}")
        return {
            "image_path": image_path,
            "question": question,
            "result": []
        }
    
    # Build command
    cmd = [
        "python", "tools/demo.py",
        "--img", full_image_path,
        "--expression", question,
        "--config", config_file,
        "--checkpoint", checkpoint_file,
        "--branch", "decoder",
        "--output-coords"
    ]
    
    try:
        # Set environment variable for GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "2"
        
        # Run command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
        
        if result.returncode == 0:
            # Parse the coordinate output
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith("COORDINATES_JSON:"):
                    json_str = line.replace("COORDINATES_JSON:", "").strip()
                    coord_data = json.loads(json_str)
                    
                    # Return with original format from VG-RS-question.json
                    return {
                        "image_path": image_path,  # Keep original format
                        "question": question,      # Keep original format
                        "result": coord_data["result"]
                    }
        
        # If no coordinates found or error occurred
        print(f"Warning: No coordinates found for {image_path} - {question}")
        return {
            "image_path": image_path,
            "question": question,
            "result": []
        }
        
    except subprocess.TimeoutExpired:
        print(f"Timeout processing {image_path} - {question}")
        return {
            "image_path": image_path,
            "question": question,
            "result": []
        }
    except Exception as e:
        print(f"Error processing {image_path} - {question}: {e}")
        return {
            "image_path": image_path,
            "question": question,
            "result": []
        }

def write_result_to_file(result, output_file, is_first=False):
    """Write a single result to JSON file incrementally"""
    import os
    
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
    
    # Statistics
    processed_count = 0
    successful_count = 0
    
    for i, item in enumerate(tqdm(dataset, desc="Processing items")):
        try:
            image_path = item["image_path"]
            question = item["question"]
            
            # Process single item
            result = process_single_item(image_path, question, config_file, checkpoint_file)
            
            # Write result immediately to file
            write_result_to_file(result, output_file, is_first=(i == 0))
            
            # Update statistics
            processed_count += 1
            if result["result"]:
                successful_count += 1
            
            # Print progress every 100 items
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{len(dataset)} | Success rate: {successful_count}/{processed_count} ({successful_count/processed_count*100:.1f}%)")
                
        except Exception as e:
            print(f"Unexpected error processing item {i}: {e}")
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

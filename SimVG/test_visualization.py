#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont
import json

def test_visualization():
    # Load visualization info
    with open('visualization_info_final.json', 'r') as f:
        viz_info = json.load(f)
    
    # Test first item
    if len(viz_info) > 0:
        item = viz_info[0]
        print(f"Testing visualization for: {item['question']}")
        print(f"Bbox: {item['bbox']}")
        
        # Load original image
        image_path = "/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-images/24_0_106_0000010_250128.jpg"
        
        if os.path.exists(image_path):
            image = Image.open(image_path)
            print(f"Original image size: {image.size}")
            
            draw = ImageDraw.Draw(image)
            
            # Get bbox coordinates
            x1, y1 = item['bbox'][0]
            x2, y2 = item['bbox'][1]
            
            print(f"Drawing bbox: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Draw thick red rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            
            # Draw text with background
            question = item['question']
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            # Draw text background
            text_bbox = draw.textbbox((x1, y1-40), question, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((x1, y1-40), question, fill="white", font=font)
            
            # Save test image
            output_path = "test_visualization_with_bbox.jpg"
            image.save(output_path)
            print(f"Test visualization saved to: {output_path}")
            
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    test_visualization()

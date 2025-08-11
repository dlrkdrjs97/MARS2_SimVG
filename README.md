# MARS2: Multimodal Reasoning Competition Track1

This repository contains the implementation of MARS2, a multi-modal analysis and reasoning system that integrates SimVG for visual grounding tasks.

## 📁 Project Structure

```
mars2/
├── SimVG/                          # SimVG visual grounding framework
│   ├── configs/                    # Configuration files
│   ├── pretrain_weights/           # Pre-trained model weights
│   ├── simvg/                      # Core SimVG implementation
│   ├── tools/                      # Utility scripts
│   ├── eval_for_MARS2_SimVG.py    # Main evaluation script for MARS2
│   └── requirements.txt            # Python dependencies
├── dataset/                        # Dataset files
│   ├── VG-RS-question.json        # Visual grounding questions
│   └── VG-RS-images/              # Image files (extract from .rar)
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0.0+

### Setup Instructions

1. **Clone and setup SimVG** (required for visual grounding):
   ```bash
   # SimVG and its dependencies must be installed first
   # Refer to: https://github.com/Dmmm1997/SimVG
   
   # Install DETRex (required by SimVG)
   pip install -e detectron2
   pip install -e .
   
   # Install SimVG
   pip install -e .
   ```

2. **Install additional dependencies**:
   ```bash
   cd SimVG
   pip install -r requirements.txt
   pip install grad-cam  # For visualization
   ```

3. **Download pre-trained weights**:
   - Place `det_best.pth` in `SimVG/pretrain_weights/`
   - Place `beit3_large_patch16_224.pth` in `SimVG/pretrain_weights/`
   - Place `beit3.spm` in `SimVG/pretrain_weights/`

## 🔧 Modifications to SimVG

### Coordinate Output Enhancement

The original SimVG project has been modified to output bounding box coordinates in addition to visualization:

- **Original**: Only visual output with bounding boxes drawn on images
- **Modified**: JSON output with precise coordinates `[[x1, y1], [x2, y2]]` (top-left, bottom-right)
- **Error Handling**: When inference fails, returns full image dimensions as bounding box
- **Format**: Maintains original `image_path` and `question` format from input JSON

### Key Changes

- `tools/demo.py`: Added `--output-coords` flag for coordinate extraction
- `simvg/core/utils.py`: Enhanced coordinate extraction functionality
- Error handling with fallback to full image bounding box

## 📊 Evaluation Script: `eval_for_MARS2_SimVG.py`

### Purpose
Batch processing script for evaluating visual grounding on large datasets with coordinate output and visualization.

### Features
- **GPU Acceleration**: Optimized for CUDA GPU usage
- **Batch Processing**: Handles large datasets efficiently
- **Incremental Saving**: Writes results to JSON file progressively
- **Error Resilience**: Continues processing even when individual items fail
- **Dual Output**: Both coordinate extraction and visualization
- **Progress Tracking**: Real-time progress monitoring

### Input Format
```json
[
  {
    "image_path": "images\\example.jpg",
    "question": "What object is next to the red car?"
  }
]
```

### Output Format
```json
[
  {
    "image_path": "images\\example.jpg",
    "question": "What object is next to the red car?",
    "result": [[x1, y1], [x2, y2]]
  }
]
```

## 🎯 Usage

### Basic Evaluation

```bash
cd SimVG
CUDA_VISIBLE_DEVICES=0 python eval_for_MARS2_SimVG.py
```

### Configuration

- **GPU Selection**: Set `CUDA_VISIBLE_DEVICES` for specific GPU
- **Dataset Path**: Modify `dataset_path` in script for custom datasets
- **Output Directory**: Results saved to `vg_rs_batch_results.json`
- **Visualization**: Images with bounding boxes saved to `visualize/results/`

### Expected Output

- **JSON Results**: `vg_rs_batch_results.json` with coordinates
- **Visual Images**: `visualize/results/` with bounding boxes drawn
- **Progress Logs**: Real-time processing status and statistics



## 📚 References

- **SimVG Paper**: [arXiv:2409.17531](https://arxiv.org/abs/2409.17531)
- **SimVG Repository**: [https://github.com/Dmmm1997/SimVG](https://github.com/Dmmm1997/SimVG)
- **DETRex**: Required dependency for SimVG

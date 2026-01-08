# labeling_tool<img width="377" height="670" alt="Ekran görüntüsü 2026-01-08 184721" src="https://github.com/user-attachments/assets/d466ca3b-920b-4dcb-b81a-b8615688008a" />


A lightweight OpenCV-based annotation tool designed for fast **GPR radargram** labeling.  
It generates **YOLO-format** labels by clicking on hyperbola apex points and automatically creating fixed-size bounding boxes.

## Why this tool?
General-purpose labeling tools can be slow for GPR radargrams because:
- Targets are ambiguous (hyperbola vs clutter/noise), so decisions take time.
- Drawing precise boxes manually adds unnecessary friction.
- Iterative correction (delete a wrong box) must be instant.

This tool optimizes the workflow for GPR:
- **Left click** to add a label (fixed-size bbox around the clicked point)
- **Right click** to delete the bbox under the cursor
- Save labels directly in **YOLO** format

## Features
- Fast point-and-click labeling for radargrams
- Fixed-size bounding boxes using `margin_x` and `margin_y`
- YOLO label export (`class x_center y_center width height`, normalized 0–1)
- Move labeled images into a clean dataset structure:
  - `labeled_images/images`
  - `labeled_images/labels`
- Optional side-by-side reference windows (if available):
  - ground-truth view
  - original radargram view
- Round-robin image ordering (balanced sampling across project/level groups)

## Folder Structure

Expected input/output structure (default):

labeling_tool/
  labeling.py
  README.md
  radagram_images/ # INPUT: images to label
  labeled_images/ # OUTPUT: created automatically
    images/
    labels/
  sgy_dataset/

## Installation

### Requirements
- Python 3.9+ recommended
- OpenCV

Install dependencies:

pip install opencv-python 


### Controls (Hotkeys & Mouse):

    Left Click: add a bounding box centered at click point

    Right Click: delete bbox under cursor

    R: reset (clear all labels for current image)

    Enter: save YOLO labels + move image to output folder

    D: skip image (no move)

    A: go back to previous image

    Q / ESC: quit


### Output Format (YOLO):

Each labeled image gets a .txt file with one line per object:
<class_id> <x_center> <y_center> <width> <height>

All values are normalized to [0, 1]

Default:

class_id = 0 (hyperbola)

### Configuration

Edit these variables:

margin_x, margin_y : bbox half-size in pixels

class_id : YOLO class id

MAX_IMAGES : stop after labeling N images (optional)

input/output folder names (if your structure differs)

### Common Pitfalls

1) Resize & Label Misalignment

If you resize images for display, ensure labels match the image size used for training.
Best practice: train on the same images that are moved to labeled_images_opencv/images.

2) File Naming Assumptions

The script may filter images by suffix (e.g. _processed.png).
If your images use different naming, adjust the filtering logic.

### Author

Onur Mefut

import cv2
import os
import glob
import shutil
import re
from collections import defaultdict

# ============================================================
# Settings
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
image_folder = os.path.join(project_root, "radargram_images")
output_folder = os.path.join(project_root, "labeled_images")
sgy_dataset_dir = os.path.join(project_root, "sgy_dataset")

# Supported image extensions
image_extensions = ["*.png", "*.jpg", "*.jpeg"]

# Bounding box half-size around each click (in pixels)
margin_x = 35
margin_y = 20

# YOLO class id (single class: hyperbola)
class_id = 0

# Maximum number of images to label in one session
MAX_IMAGES = 300

# Output folders for YOLO training data
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)


# ============================================================
# Parsing helpers: project / level / path
# ============================================================

def parse_image_meta(img_path):
    """
    Parse (project_id, level, path_num) from a filename like:

        01_01_01_1_Radargrams__Path1_background_removal.png
        04_04_1_Radargrams__Path3_background_removal.png
        013_013_1_Radargrams__Path12_background_removal.png

    Returns:
        (project_id: int, level: int, path_num: int)
        or None if parsing fails.
    """
    base = os.path.basename(img_path)
    stem, _ext = os.path.splitext(base)

    # Strip processing suffix
    for suf in ("_processed", "_original", "_background_removal", "_envelope"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    # Left: directory structure   Right: file stem (PathN, ...)
    if "__" in stem:
        left, right = stem.split("__", 1)
    else:
        left, right = stem, ""

    # Numbers on the left: first = project, last = level
    nums = re.findall(r"\d+", left)
    if not nums:
        return None

    project_id = int(nums[0])
    level = int(nums[-1])

    # Path number from right ("Path1", "Path2", ...)
    path_num = 1
    m = re.search(r"Path(\d+)", right)
    if m:
        path_num = int(m.group(1))

    return project_id, level, path_num

def get_ground_truth_path_for_image(img_path):
    """
    Given a radar_images PNG filename like
        01_01_01_9_Radargrams__Path8_processed.png
    reconstruct the path to the corresponding ground-truth.png, e.g.
        sgy_dataset/01/01/01.9/ground-truth.png
    """
    base = os.path.basename(img_path)
    stem, _ = os.path.splitext(base)

    # Strip variant suffix (_processed, _original, _background_removal, _envelope)
    for suffix in ("_processed", "_original", "_background_removal", "_envelope"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    # Expect something like: 01_01_01_9_Radargrams__Path8
    if "__" not in stem:
        return None

    left, _ = stem.split("__", 1)
    parts = left.split("_")
    if len(parts) < 5:
        return None

    folder1, folder2, lvl_major, lvl_minor, radargrams_token = parts[:5]
    level_folder = f"{lvl_major}.{lvl_minor}"   # e.g. "01.9" or "10.1"

    gt_path = os.path.join(
        sgy_dataset_dir,
        folder1,
        folder2,
        level_folder,
        "ground-truth.png",
    )

    if os.path.exists(gt_path):
        return gt_path
    return None

def get_original_image_path(img_path):
    """
    Given a processed PNG like:
        01_01_01_9_Radargrams__Path8_processed.png
    return the corresponding _original.png in the SAME folder, e.g.:
        01_01_01_9_Radargrams__Path8_original.png

    Returns:
        full path to _original.png if it exists, else None
    """
    dir_name = os.path.dirname(img_path)
    base = os.path.basename(img_path)
    stem, ext = os.path.splitext(base)

    suffix = "_background_removal"
    if stem.endswith(suffix):
        orig_stem = stem[:-len(suffix)] + "_original"
        orig_path = os.path.join(dir_name, orig_stem + ext)
        if os.path.exists(orig_path):
            return orig_path
    elif stem.endswith("_processed"):
        orig_stem = stem[:-len("_processed")] + "_original"
        orig_path = os.path.join(dir_name, orig_stem + ext)
        if os.path.exists(orig_path):
            return orig_path

    return None

# ============================================================
# Point-in-Box Detection for Right-Click Delete
# ============================================================

def find_box_at_position(x, y, points):
    """
    Find which bounding box (if any) contains the given mouse position.
    
    This function checks all existing bounding boxes to see if the mouse
    cursor is inside any of them. It returns the index of the box that
    contains the cursor, prioritizing boxes that were added more recently
    (later in the list) if there are overlapping boxes.
    
    Args:
        x: Mouse x coordinate in pixels
        y: Mouse y coordinate in pixels
        points: List of (center_x, center_y) tuples
    
    Returns:
        Index of the box containing the point, or None if no box contains it
    """
    # Check boxes in reverse order (most recent first)
    # This way if boxes overlap, we delete the one on top
    for idx in range(len(points) - 1, -1, -1):
        center_x, center_y = points[idx]
        
        # Calculate the bounding box boundaries
        x1 = center_x - margin_x
        y1 = center_y - margin_y
        x2 = center_x + margin_x
        y2 = center_y + margin_y
        
        # Check if the mouse position is inside this box
        if x1 <= x <= x2 and y1 <= y <= y2:
            return idx
    
    return None


# ============================================================
# Collect and order images (multi-pass round robin)
# ============================================================

def build_ordered_image_list():
    # Collect all background_removal images
    raw_image_files = []
    for ext in image_extensions:
        pattern = os.path.join(image_folder, ext)
        for f in glob.glob(pattern):
            if "_background_removal.png" in f:
                raw_image_files.append(f)

    if not raw_image_files:
        print("No _background_removal images found in:", image_folder)
        return [], {}

    groups = defaultdict(list)  # (project_id, level) -> list of (path_num, img_path)
    projects_set = set()
    levels_set = set()

    for img_path in raw_image_files:
        meta = parse_image_meta(img_path)
        if meta is None:
            print(f"[WARN] Could not parse meta from '{os.path.basename(img_path)}', skipping.")
            continue

        project_id, level, path_num = meta
        projects_set.add(project_id)
        levels_set.add(level)
        groups[(project_id, level)].append((path_num, img_path))

    if not groups:
        print("No parsable images after meta extraction.")
        return [], {}

    # Sort each group by path number so that:
    #   pass 1 -> Path1 from each group
    #   pass 2 -> Path2 from each group, etc.
    for key in groups:
        groups[key].sort(key=lambda t: t[0])  # sort by path_num

    projects_sorted = sorted(projects_set)
    levels_sorted = sorted(levels_set)

    # Keep a copy of original sizes for diversity summary
    original_group_sizes = {key: len(val) for key, val in groups.items()}

    ordered_image_files = []

    # Multi-pass round robin:
    # Pass 1: one image from each non-empty group
    # Pass 2: again one image from each non-empty group, etc.
    while True:
        added_in_this_pass = False

        for level in levels_sorted:
            for project_id in projects_sorted:
                key = (project_id, level)
                if key in groups and groups[key]:
                    _path_num, img_path = groups[key].pop(0)
                    ordered_image_files.append(img_path)
                    added_in_this_pass = True

                    if MAX_IMAGES is not None and len(ordered_image_files) >= MAX_IMAGES:
                        break
            if MAX_IMAGES is not None and len(ordered_image_files) >= MAX_IMAGES:
                break

        # If we didn't add anything this pass, all groups are empty
        if not added_in_this_pass:
            break

        if MAX_IMAGES is not None and len(ordered_image_files) >= MAX_IMAGES:
            break

    if not ordered_image_files:
        print("No images selected after ordering/filtering.")
        return [], original_group_sizes

    print("ROUND-ROBIN SAMPLING SUMMARY:/n")
    print(f"Total background_removal images available: {len(raw_image_files)}")
    print(f"Total images selected for this labeling run: {len(ordered_image_files)}")
    print(f"MAX_IMAGES setting: {MAX_IMAGES}")

    fully_used_keys = []

    # After selection, 'groups[key]' contains leftovers
    for (project_id, level) in sorted(original_group_sizes.keys()):
        total = original_group_sizes[(project_id, level)]
        leftover = len(groups.get((project_id, level), []))
        used = total - leftover
        fully_used = (used == total and total > 0)
        if fully_used:
            fully_used_keys.append((project_id, level))

        print(f"   {project_id:7d}  {level:5d}   {used:4d} / {total:4d}      {'YES' if fully_used else 'no'}")

    if fully_used_keys:
        print("\nProject-level combinations where ALL images were used in this run:")
        for project_id, level in fully_used_keys:
            print(f"  - project {project_id}, level {level}")
    else:
        print("\nNo (project, level) combination had all its images used in this run.")

    print("\nFirst 20 images in labeling order:")
    for i, img in enumerate(ordered_image_files[:20]):
        meta = parse_image_meta(img)
        if meta is not None:
            project_id, level, path_num = meta
            print(f"  {i+1:2d}. P{project_id:02d} L{level:02d} Path{path_num:02d}  |  {os.path.basename(img)}")
        else:
            print(f"  {i+1:2d}. (unparsed) {os.path.basename(img)}")

    print("=" * 70 + "\n")

    return ordered_image_files, original_group_sizes


# ============================================================
# YOLO label saving + image moving
# ============================================================

def save_yolo_labels(img_path, points):
    """
    Save YOLO-format labels for a single image based on clicked points.
    Each point becomes a bounding box centered at (x, y) with margins
    margin_x and margin_y.
    """
    if not points:
        return None

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read image for labeling: {img_path}")
        return None

    h, w = img.shape[:2]

    base = os.path.basename(img_path)
    stem, _ext = os.path.splitext(base)
    txt_path = os.path.join(output_folder, "labels", stem + ".txt")

    lines = []
    for (x, y) in points:
        # Bounding box coordinates in absolute pixels
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w - 1, x + margin_x)
        y2 = min(h - 1, y + margin_y)

        # YOLO normalized format (center_x, center_y, width, height)
        center_x = ((x1 + x2) / 2.0) / w
        center_y = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bw:.6f} {bh:.6f}\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[LABEL] Saved {len(points)} boxes to {txt_path}")
    return txt_path


def move_image_to_output(img_path):
    """
    Move the labeled image into labeled_images/images.
    """
    img_basename = os.path.basename(img_path)
    dest_img = os.path.join(output_folder, "images", img_basename)

    # If an image with the same name already exists, overwrite it
    if os.path.exists(dest_img):
        os.remove(dest_img)

    shutil.move(img_path, dest_img)
    print(f"[MOVE] {img_basename} → labeled_images/images/")

# ============================================================
# Main labeling loop
# ============================================================

window_name = "Hyperbola Labeling Tool"
current_points = []
current_base_img = None
current_display_img = None


def redraw_display():
    """
    Redraw the display image with all current points overlaid:
    - green dot
    - index number
    - yellow bounding box with margin_x/margin_y
    """
    global current_display_img, current_base_img, current_points
    if current_base_img is None:
        return

    current_display_img = current_base_img.copy()
    h, w = current_base_img.shape[:2]

    for idx, (x, y) in enumerate(current_points):
        # Draw point
        cv2.circle(current_display_img, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(
            current_display_img,
            str(idx),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

        # Draw bounding box (same logic as original script)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w - 1, x + margin_x)
        y2 = min(h - 1, y + margin_y)
        cv2.rectangle(current_display_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imshow(window_name, current_display_img)


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback for placing and removing points.
    Controls:
    - Left click: Add a new point (with bounding box) at cursor position
    - Right click: Delete the SPECIFIC box under the cursor
    """
    global current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click: add new point
        current_points.append((x, y))
        redraw_display()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click: delete the box at this position
        box_idx = find_box_at_position(x, y, current_points)
        
        if box_idx is not None:
            deleted_point = current_points.pop(box_idx)
            print(f"[DELETE] Removed box #{box_idx} at position {deleted_point}")
            redraw_display()
        else:
            print("[DELETE] No bounding box found at cursor position")

def main():
    global current_base_img, current_display_img, current_points

    image_files, _ = build_ordered_image_list()
    if not image_files:
        return

    # Window size will always match image size
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    idx = 0
    num_images = len(image_files)
    labeled_count = 0

    print("Controls:")
    print("  - Left click: add point (with bounding box)")
    print("  - Right click: delete box under cursor")
    print("  - ENTER: save labels & move image, go to next")
    print("  - D or Right Arrow: skip image without saving")
    print("  - A or Left Arrow: go back to previous image")
    print("  - R: reset points on current image")
    print("  - Q or ESC: quit\n")

    while 0 <= idx < num_images:
        img_path = image_files[idx]
        current_base_img = cv2.imread(img_path)

        if current_base_img is None:
            print(f"[WARN] Could not read image: {img_path}, skipping.")
            idx += 1
            continue

        base_name = os.path.basename(img_path)
        cv2.setWindowTitle(window_name, f"{window_name} - {base_name}")

        gt_path = get_ground_truth_path_for_image(img_path)
        gt_window_name = "Ground Truth"


        if gt_path:
            gt_img = cv2.imread(gt_path)
            if gt_img is not None:
                cv2.imshow(gt_window_name, gt_img)
            else:
                print(f"[WARN] Could not read ground-truth image: {gt_path}")
                try:
                    cv2.destroyWindow(gt_window_name)
                except cv2.error:
                    pass
        else:
            # No GT for this image → close any previous GT window
            try:
                cv2.destroyWindow(gt_window_name)
            except cv2.error:
                pass

         # show corresponding _original.png next to processed image
        orig_window_name = "Original Radargram"
        orig_path = get_original_image_path(img_path)

        if orig_path:
            orig_img = cv2.imread(orig_path)
            if orig_img is not None:
                cv2.imshow(orig_window_name, orig_img)
                try:
                    orig_base = os.path.basename(orig_path)
                    cv2.setWindowTitle(orig_window_name, f"{orig_window_name} - {orig_base}")
                except cv2.error:
                    pass
            else:
                print(f"[WARN] Could not read original image: {orig_path}")
                try:
                    cv2.destroyWindow(orig_window_name)
                except cv2.error:
                    pass
        else:
            # No _original image for this processed image, close any previous window
            try:
                cv2.destroyWindow(orig_window_name)
            except cv2.error:
                pass

        current_points = []
        redraw_display()

        print(f"\n=== Image {idx+1}/{num_images} ===")
        print(f"{img_path}")
        print(f"Currently labeled: {labeled_count}/{num_images} images.")
        print("Mark hyperbolas, then press ENTER to save or D to skip.")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 13:  # ENTER
                if current_points:
                    txt_path = save_yolo_labels(img_path, current_points)
                    move_image_to_output(img_path)
                    labeled_count += 1
                    print(f"[PROGRESS] Labeled {labeled_count}/{num_images} images so far.")
                else:
                    print("[INFO] No points placed, image left in original folder.")
                idx += 1
                break

            elif key in (ord('d'), 83):  # 'd' or Right Arrow -> skip
                print("[SKIP] Skipping image without saving.")
                idx += 1
                break

            elif key in (ord('a'), 81):  # 'a' or Left Arrow -> previous
                if idx > 0:
                    print("[NAV] Going back to previous image.")
                    idx -= 1
                    break
                else:
                    print("[NAV] Already at first image; can't go back.")

            elif key == ord('r'):
                print("[RESET] Cleared all points on this image.")
                current_points = []
                redraw_display()

            elif key in (27, ord('q')):  # ESC or 'q'
                print("[EXIT] Exiting labeling tool.")
                print(f"[SUMMARY] Labeled {labeled_count}/{num_images} images this run.")
                cv2.destroyAllWindows()
                return

    print("\n[DONE] No more images in the list.")
    print(f"[SUMMARY] Labeled {labeled_count}/{num_images} images this run.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Complete image association and renaming script
Integrates the functionality of dir2txt.py, associate.py, and associate_rename.py

Features:
1. Generate timestamp files from color and depth folders
2. Associate timestamps of color and depth images
3. Rename files based on association results
4. Handle unmatched images
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple


def read_file_list(filename: str) -> Dict[float, List[str]]:
    """
    Read data from timestamp file

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list_data = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines
                 if len(line) > 0 and line[0] != "#"]
    list_data = [(float(l[0]), l[1:]) for l in list_data if len(l) > 1]
    return dict(list_data)


def associate(first_list: Dict[float, List[str]], second_list: Dict[float, List[str]],
              offset: float = 0.0, max_difference: float = 0.02) -> List[Tuple[float, float]]:
    """
    Associate two dictionaries of (stamp, data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1), (stamp2,data2))
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


def generate_timestamp_files(color_dir: str, depth_dir: str, output_dir: str) -> Tuple[str, str]:
    """
    Generate timestamp files from folders

    Args:
        color_dir: color image folder path
        depth_dir: depth image folder path
        output_dir: output directory

    Returns:
        (color_txt_path, depth_txt_path)
    """
    # Get and sort files
    color_files = [f for f in os.listdir(color_dir) if f.endswith('.png')]
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]

    # Sort by timestamp (assuming filenames are timestamps)
    color_files.sort(key=lambda x: float(x[:-4]))
    depth_files.sort(key=lambda x: float(x[:-4]))

    # Generate timestamp files
    color_txt = os.path.join(output_dir, "color.txt")
    depth_txt = os.path.join(output_dir, "depth.txt")

    with open(color_txt, "w") as f:
        for filename in color_files:
            timestamp = filename[:-4]  # remove .png extension
            f.write(f"{timestamp} color/{filename}\n")

    with open(depth_txt, "w") as f:
        for filename in depth_files:
            timestamp = filename[:-4]  # remove .png extension
            f.write(f"{timestamp} depth/{filename}\n")

    print(f"Generated timestamp files: {color_txt}, {depth_txt}")
    return color_txt, depth_txt


def process_association(color_txt: str, depth_txt: str, output_dir: str,
                       offset: float = 0.0, max_difference: float = 0.02) -> str:
    """
    Process timestamp association

    Args:
        color_txt: color timestamp file path
        depth_txt: depth timestamp file path
        output_dir: output directory
        offset: time offset
        max_difference: maximum allowed time difference

    Returns:
        associations_txt_path
    """
    # Read timestamp files
    color_list = read_file_list(color_txt)
    depth_list = read_file_list(depth_txt)

    print(f"Number of color images: {len(color_list)}")
    print(f"Number of depth images: {len(depth_list)}")

    # Associate timestamps
    matches = associate(color_list, depth_list, offset, max_difference)

    # Save association results
    associations_txt = os.path.join(output_dir, "associations.txt")
    with open(associations_txt, "w") as f:
        for color_stamp, depth_stamp in matches:
            color_data = color_list[color_stamp]
            depth_data = depth_list[depth_stamp]
            f.write(f"{color_stamp} {' '.join(color_data)} {depth_stamp} {' '.join(depth_data)}\n")

    print(f"Successfully associated {len(matches)} pairs of images")
    print(f"Unmatched color images: {len(color_list) - len(matches)}")
    print(f"Unmatched depth images: {len(depth_list) - len(matches)}")

    return associations_txt


def rename_files(associations_txt: str, output_dir: str, handle_unmatched: str = "keep"):
    """
    Rename files based on association results

    Args:
        associations_txt: association results file path
        output_dir: output directory
        handle_unmatched: how to handle unmatched images ("keep", "delete", "move")
    """
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")

    # Read association results
    association = np.loadtxt(associations_txt, dtype=str)
    print(f"Association results shape: {association.shape}")

    # Process pose file (if exists)
    pose_file = os.path.join(output_dir, 'poses.txt')
    if os.path.exists(pose_file):
        poses_quad = np.loadtxt(pose_file)
        index = np.arange(poses_quad.shape[0])
        poses_quad[:, 0] = index
        np.savetxt(os.path.join(output_dir, 'pose.txt'), poses_quad)
        print("Renamed poses.txt -> pose.txt")
    else:
        print("poses.txt file not found")

    # Get all original files
    all_color_files = set(os.listdir(color_dir))
    all_depth_files = set(os.listdir(depth_dir))

    # Rename matched files
    matched_color = set()
    matched_depth = set()

    for i in range(association.shape[0]):
        tmp = association[i]
        rgb_filename = tmp[1]  # color/filename.png
        depth_filename = tmp[3]  # depth/filename.png

        rgb_basename = os.path.basename(rgb_filename)
        depth_basename = os.path.basename(depth_filename)

        old_rgb = os.path.join(color_dir, rgb_basename)
        new_rgb = os.path.join(color_dir, f"{i}.png")
        old_depth = os.path.join(depth_dir, depth_basename)
        new_depth = os.path.join(depth_dir, f"{i}.png")

        if os.path.exists(old_rgb):
            os.rename(old_rgb, new_rgb)
            matched_color.add(rgb_basename)

        if os.path.exists(old_depth):
            os.rename(old_depth, new_depth)
            matched_depth.add(depth_basename)

    print(f"Renamed {len(matched_color)} color files and {len(matched_depth)} depth files")

    # Handle unmatched files
    unmatched_color = all_color_files - matched_color
    unmatched_depth = all_depth_files - matched_depth

    if handle_unmatched == "delete":
        for filename in unmatched_color:
            os.remove(os.path.join(color_dir, filename))
        for filename in unmatched_depth:
            os.remove(os.path.join(depth_dir, filename))
        print(f"Deleted {len(unmatched_color)} unmatched color files and {len(unmatched_depth)} unmatched depth files")

    elif handle_unmatched == "move":
        unmatched_dir = os.path.join(output_dir, "unmatched")
        os.makedirs(unmatched_dir, exist_ok=True)

        unmatched_color_dir = os.path.join(unmatched_dir, "color")
        unmatched_depth_dir = os.path.join(unmatched_dir, "depth")
        os.makedirs(unmatched_color_dir, exist_ok=True)
        os.makedirs(unmatched_depth_dir, exist_ok=True)

        for filename in unmatched_color:
            os.rename(os.path.join(color_dir, filename), os.path.join(unmatched_color_dir, filename))
        for filename in unmatched_depth:
            os.rename(os.path.join(depth_dir, filename), os.path.join(unmatched_depth_dir, filename))
        print(f"Moved {len(unmatched_color)} unmatched color files and {len(unmatched_depth)} unmatched depth files to {unmatched_dir}")

    else:  # keep
        print(f"Kept {len(unmatched_color)} unmatched color files and {len(unmatched_depth)} unmatched depth files")


def main():
    parser = argparse.ArgumentParser(description='Complete image association and renaming tool')
    parser.add_argument('scene_dir', help='Scene directory path (containing color and depth folders)')
    parser.add_argument('--offset', type=float, default=0.0,
                       help='Time offset (default: 0.0)')
    parser.add_argument('--max_difference', type=float, default=0.02,
                       help='Maximum allowed time difference (default: 0.02)')
    parser.add_argument('--handle_unmatched', choices=['keep', 'delete', 'move'], default='delete',
                       help='How to handle unmatched images (default: delete)')

    args = parser.parse_args()

    scene_dir = args.scene_dir.rstrip('/')
    color_dir = os.path.join(scene_dir, "color")
    depth_dir = os.path.join(scene_dir, "depth")

    # Check if directories exist
    if not os.path.exists(color_dir):
        print(f"Error: color directory does not exist: {color_dir}")
        sys.exit(1)

    if not os.path.exists(depth_dir):
        print(f"Error: depth directory does not exist: {depth_dir}")
        sys.exit(1)

    print(f"Processing scene: {scene_dir}")
    print(f"Color directory: {color_dir}")
    print(f"Depth directory: {depth_dir}")

    # Step 1: Generate timestamp files
    print("\n=== Step 1: Generate timestamp files ===")
    color_txt, depth_txt = generate_timestamp_files(color_dir, depth_dir, scene_dir)

    # Step 2: Associate timestamps
    print("\n=== Step 2: Associate timestamps ===")
    associations_txt = process_association(color_txt, depth_txt, scene_dir,
                                         args.offset, args.max_difference)

    # Step 3: Rename files
    print("\n=== Step 3: Rename files ===")
    rename_files(associations_txt, scene_dir, args.handle_unmatched)

    print("\n=== Processing completed ===")
    print(f"Results saved in: {scene_dir}")


if __name__ == '__main__':
    main()

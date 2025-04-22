import json
import numpy as np
import os

# Load the JSON file
json_path = '/Users/siddhu/PycharmProjects/LayoutEstimation/pix3d/pix3d.json'
with open(json_path, 'r') as f:
    annotations = json.load(f)

# Find the entry for our specific image
target_img = 'img/bed/0001.png'
target_entry = None
for entry in annotations:
    if entry.get('img') == target_img:
        target_entry = entry
        break

if target_entry:
    print("Found entry for image:", target_img)

    # Extract camera information
    rot_mat = np.array(target_entry.get('rot_mat'))
    trans_mat = np.array(target_entry.get('trans_mat'))
    focal_length = target_entry.get('focal_length')

    # Extract camera pitch and roll (simplified conversion)
    pitch = np.arcsin(-rot_mat[2, 0]) if abs(rot_mat[2, 0]) <= 1 else 0
    roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2]) if rot_mat[2, 2] != 0 else 0

    print("\nGround Truth Data:")
    print("Rotation Matrix:\n", rot_mat)
    print("Translation Matrix:", trans_mat)
    print("Focal Length:", focal_length)
    print("Calculated Camera Pitch (radians):", pitch)
    print("Calculated Camera Roll (radians):", roll)

    # Extract other relevant information
    img_size = target_entry.get('img_size')
    category = target_entry.get('category')
    model_path = target_entry.get('model')

    print("\nAdditional Information:")
    print("Image Size:", img_size)
    print("Category:", category)
    print("3D Model Path:", model_path)

    # Now you can compare this with the model's predictions
    print("\nTo compare with model predictions, check the HTML report generated in outputs/demo")
    print("The model should have predicted layout parameters and camera parameters")
else:
    print("Entry not found for image:", target_img)
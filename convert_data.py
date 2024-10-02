import os
import glob
import shutil
from pathlib import Path

data_path= 'data'
output_path = 'dataset'
output_video_name = 'kitchen'

rgb_images = glob.glob(os.path.join(data_path, '*.color.jpg'))
depth_images = glob.glob(os.path.join(data_path, '*.depth.png'))
output_path = os.path.join(output_path, output_video_name)
new_rgb_path = os.path.join(output_path, 'color')
new_depth_path = os.path.join(output_path, 'depth')

if not os.path.isdir(new_rgb_path):
    os.makedirs(new_rgb_path)

if not os.path.isdir(new_depth_path):
    os.makedirs(new_depth_path)

for rgb_path, depth_path in zip(rgb_images, depth_images):
    filename = Path(rgb_path).name
    middle_part = filename[6:]
    number_part = middle_part.split(".")[0] 
    new_filename = f"{int(number_part):04d}-color.jpg"    
    shutil.copy2(rgb_path, os.path.join(new_rgb_path, new_filename))

    filename = Path(depth_path).name
    middle_part = filename[6:]
    number_part = middle_part.split(".")[0]
    new_filename = f"{int(number_part):04d}-depth.png"
    shutil.copy2(depth_path, os.path.join(new_depth_path, new_filename))

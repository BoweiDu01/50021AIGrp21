import cv2
import os
from PIL import Image
import pillow_avif

def convert_avif_to_png(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".avif"):
                img_path = os.path.join(input_folder, filepath)
                print(img_path)
                img = Image.open(img_path)    
                img.save(os.path.join(output_folder, filepath.replace(".avif", ".png")))
                img = None
                os.remove(img_path)                            


input_dir = "C:/Users/ryanc/Documents/50021AIGrp21/Dataset"
output_dir = "C:/Users/ryanc/Documents/50021AIGrp21/Dataset"
convert_avif_to_png(input_dir, output_dir)
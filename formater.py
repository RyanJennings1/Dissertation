#!/usr/bin/env python3
"""
  name: formater.py
  author: Ryan Jennings
  date: 2021-08-13
"""
import csv
import os

import cv2

from typing import Dict, List
from shutil import copyfile

from matplotlib import pyplot as plt # for checking masks
from plantcv import plantcv as pcv

def main() -> None:
  """
  main method
  """
  generate_masks()

class options:
  """
  PlantCV options details class
  """
  def __init__(self, image: str) -> None:
    """
    init method

    keyword arguments:
    image: str - name of input file

    return: None
    """
    self.image = image or "./img/arabidopsis.jpg"
    self.debug = None # Set to "plot" to show each image
    self.writeimg = False
    self.outdir = "."

def generate_masks() -> None:
  """
  For each image in images
  generate a mask image for it
  and store in masks directory
  """
  dest_path: str = "masks"
  images_path: str = "images"
  threshold: int = 116
  input_images: List[str] = sorted(os.listdir(images_path))
  for image_name in input_images:
    # Code from PlantCV Watershed:
    # https://plantcv.readthedocs.io/en/stable/tutorials/watershed_segmentation_tutorial/
    pcv_args = options(image=f"{images_path}/{image_name}")
    pcv.params.debug = pcv_args.debug
    # Read in image to apply watershedding to
    ##img, path, filename = pcv.readimage(filename=pcv_args.image)
    img = cv2.imread(pcv_args.image)
    # Converting from RGB to LAB and keep green-magenta channel
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    # Set up a binary threshold image
    img_binary = pcv.threshold.binary(gray_img=a, threshold=threshold,
                                      max_value=255, object_type='dark')
    # Blur image to reduce noise
    img_binary = pcv.median_blur(gray_img=img_binary, ksize=20)
    # Overlay of mask onto image
    id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=img_binary)
    # If not objects detected, rerun with softer threshold
    while (not id_objects):
      threshold += 4
      img_binary = pcv.threshold.binary(gray_img=a, threshold=threshold,
                                        max_value=255, object_type='dark')
      # Blur image to reduce noise
      img_binary = pcv.median_blur(gray_img=img_binary, ksize=20)
      # Overlay of mask onto image
      id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=img_binary)
    # Reset threshold
    threshold = 116
    # Combine objects
    obj, mask = pcv.object_composition(img=img,
                                       contours=id_objects,
                                       hierarchy=obj_hierarchy)
    # Apply mask
    masked = pcv.apply_mask(img=img, mask=mask, mask_color="black")
    #plt.imshow(masked)
    #plt.show()
    cv2.imwrite(f"{dest_path}/{image_name}", masked)
    print(f"Wrote mask to {dest_path}/{image_name}")

def transfer_images() -> None:
  """
  Move images from 1,2,3,4,5,6,7,8 folders
  and transfer to images while keeping count
  of plant image numbers and writing truth
  values of the number of leaves to a csv file
  """
  leaf_counts: Dict[str, int] = {}
  INIT_COUNT: int = 1
  images_path: str = "1"
  image_paths: List[str] = ["1", "2", "3", "4", "5", "6", "7", "8"]
  dest_path: str = "images"

  for ipath in image_paths:
    files: List[str] = sorted(os.listdir(ipath))
    dest_files: List[str] = sorted(os.listdir(dest_path))
    if dest_files:
      # Get last file number in directory and increment
      INIT_COUNT = int(dest_files[-1][5:].split(".png")[0]) + 1
    for f in files:
      format_count: str = f"{INIT_COUNT:05}"
      INIT_COUNT += 1 # Keep global count independent of run
      copyfile(f"{ipath}/{f}", f"{dest_path}/plant{format_count}.png")
      leaf_counts[f"plant{format_count}.png"] = int(ipath)
      print(f"Copied {f} to {dest_path}/plant{format_count}.png")

  fieldnames: List[str] = ["filename", "count"]
  with open("truth.csv", "w") as truth:
    csv_writer = csv.DictWriter(truth, fieldnames=fieldnames)
    csv_writer.writeheader()
    for name, count in leaf_counts.items():
        csv_writer.writerow({'filename': name, 'count': count})

if __name__ == "__main__":
  main()

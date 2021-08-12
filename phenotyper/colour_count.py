#!/usr/bin/env python3
"""
  name: colour_count.py
  author: Ryan Jennings
  date: 2021-08-12
"""
import cv2

import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from typing import Dict, List, Union

from sklearn.cluster import KMeans

from plantcv import plantcv as pcv

from phenotyper.pcv_options import options

def colour_count(args: Dict[str, Union[str, bool]]) -> Dict[str, float]:
  """
  Find colour counts for colour hexs from image

  Code modified from: https://github.com/kb22/Color-Identification-using-Machine-Learning/blob/master/Color%20Identification%20using%20Machine%20Learning.ipynb
  """
  masked_image = get_mask_of_image(args.input)
  true_colour_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
  return get_colours(true_colour_masked, 8, True)

def get_mask_of_image(filename: str):
  """
  Get mask given an image, cropping out background
  to keep just the leaves
  """
  # Code from PlantCV Watershed:
  # https://plantcv.readthedocs.io/en/stable/tutorials/watershed_segmentation_tutorial/
  pcv_args = options(image=filename)
  pcv.params.debug = pcv_args.debug
  # Read in image to apply watershedding to
  ##img, path, filename = pcv.readimage(filename=pcv_args.image)
  img = cv2.imread(pcv_args.image)
  # Converting from RGB to LAB and keep green-magenta channel
  a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
  # Set up a binary threshold image
  img_binary = pcv.threshold.binary(gray_img=a, threshold=116,
                                    max_value=255, object_type='dark')
  # Blur image to reduce noise
  img_binary = pcv.median_blur(gray_img=img_binary, ksize=20)
  # Overlay of mask onto image
  id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=img_binary)
  # Combine objects
  obj, mask = pcv.object_composition(img=img,
                                     contours=id_objects,
                                     hierarchy=obj_hierarchy)
  # Apply mask
  return pcv.apply_mask(img=img, mask=mask, mask_color="black")

def get_image(image_path):
    c_image = cv2.imread(image_path)
    c_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2RGB)
    return c_image

def rgb2hex(color) -> str:
    """
    Convert a RGB number to a hex colour string
    """
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colours(image, number_of_colours, show_chart):
  modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
  modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

  clf = KMeans(n_clusters=number_of_colours + 1) # +1 as black will be removed
  labels = clf.fit_predict(modified_image)

  counts = Counter(labels)
  # sort to ensure correct colour percentage
  counts = dict(sorted(counts.items()))

  center_colours = clf.cluster_centers_
  # We get ordered colours by iterating through the keys
  ordered_colours = [center_colours[i] for i in counts.keys()]
  hex_colours = [rgb2hex(ordered_colours[i]) for i in counts.keys()]
  rgb_colours = [ordered_colours[i] for i in counts.keys()]

  # Remove black from counts by checking for the row with all values < 1 rgb
  for i, col in enumerate(ordered_colours):
    if np.all(col < 1):
      del rgb_colours[i]
      del hex_colours[i]
      counts.pop(i)
      break
  # Reorder counts so no sequential gap that would give a KeyError later
  counts = {i: c[1] for i, c in enumerate(list(counts.items()))}

  if (show_chart):
    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=hex_colours, colors=hex_colours)
    plt.show()

  def get_percent(hex_str: str) -> float:
    count_values: List[int] = list(counts.values())[:number_of_colours]
    total: int = sum([c for c in count_values])
    hex_index: int = hex_colours.index(hex_str)
    # counts[hex_index] cannot be 0 as Counter would not
    # log 0 occurances so no need for save division check
    return (counts[hex_index]/total) * 100
  colour_data: Dict[str, float] = {h: get_percent(h) for h in hex_colours}
  return colour_data
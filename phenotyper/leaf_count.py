#!/usr/bin/env python3
"""
  name: leaf_count.py
  author: Ryan Jennings
  date: 2021-08-11
"""
import json

import cv2

import numpy as np
import tensorflow as tf

from collections import Counter
from itertools import takewhile
from typing import Dict, Union

from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from tensorflow import keras

from plantcv import plantcv as pcv

from phenotyper.pcv_options import options

JSON_TYPE = Dict[str, str] # shorthand

def leaf_count(args: Dict[str, Union[bool, str]],
               model: str = "PLANTCV") -> JSON_TYPE:
  """
  Find leaf count phenotype data from various watershed
  methods

  keyword arguments:
  args: Dict[str, Union[bool, str]] - CLI arguments passed in
  model: str - which model to use for watershedding
               PLANTCV - use PlantCV built in
               OPENCV - use OpenCV

  return: JSON_TYPE - results from watershedding as JSON
  """
  threshold: int = 116
  # Code from PlantCV Watershed:
  # https://plantcv.readthedocs.io/en/stable/tutorials/watershed_segmentation_tutorial/
  pcv_args = options(image=args.input)
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

  # Using OpenCV for thresholding
  if model == "OPENCV":
    return opencv_watershed(masked, mask)

  # Using ML model for thresholding
  if model == "ML":
    mask_path: str = "temp/mask.png"
    cv2.imwrite(mask_path, masked)
    print("masked: ", masked)
    return ml_watershed(pcv_args.image, mask_path)

  # Using PlantCV watershed functionality
  return plantcv_watershed(masked, mask)


def plantcv_watershed(masked, mask) -> JSON_TYPE:
  """
  Use PlantCV built in method for watershedding

  keyword arguments:
  masked - Masked image area
  mask - Mask of image

  return: JSON_TYPE - JSON data of segmentation results
  """
  segementation_results: str = "segmentation_results.json"
  analysis_images = pcv.watershed_segmentation(rgb_img=masked,
                                               mask=mask,
                                               distance=35,
                                               label="default")
  # Save analysis results
  pcv.outputs.save_results(filename=segementation_results)
  with open(segementation_results, 'r') as results:
      return {
        "count": json.loads(results.read())['observations']['default']['estimated_object_count']['value'],
      }

def opencv_watershed(masked, mask) -> JSON_TYPE:
  """
  Use OpenCV for watershedding
  PlantCV is used for initial processing

  keyword arguments:
  masked - Masked image area
  mask - Mask of image

  return: JSON_TYPE - JSON data of segmentation results
  """
  # For code and detailed explanation see:
  # http://datahacker.rs/007-opencv-projects-image-segmentation-with-watershed-algorithm/
  threshold: int = 30
  gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
  ret, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
  # Noise removal
  kernel = np.ones((3), np.uint8)
  opening_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=9)
  # Noise removal
  closing_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=4)
  dist_transform = cv2.distanceTransform(255 - closing_img, cv2.DIST_L2, 3)
  local_max_location = peak_local_max(dist_transform, min_distance=1, indices=True)

  n_increases: int = 0
  while local_max_location.shape[0] < 30 and n_increases < 15:
    threshold += 20
    ret, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Noise removal
    kernel = np.ones((3), np.uint8)
    opening_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=9)
    # Noise removal
    closing_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=4)
    dist_transform = cv2.distanceTransform(255 - closing_img, cv2.DIST_L2, 3)
    local_max_location = peak_local_max(dist_transform, min_distance=1, indices=True)
    n_increases += 1
  # Reset threshold
  threshold = 30

  num_clusters: int = 30
  if n_increases >= 15:
    num_clusters = local_max_location.shape[0]
  kmeans = KMeans(n_clusters=num_clusters)
  # If local_max_location size is 0, return 0 predictions
  if not local_max_location.size:
    return {
      "count": 0
    }
  kmeans.fit(local_max_location)
  local_max_location = kmeans.cluster_centers_.copy()
  # Kmeans is returning a float data type so we need to convert it to an int. 
  local_max_location = local_max_location.astype(int)
  dist_transform_copy = dist_transform.copy()
  for i in range(local_max_location.shape[0]):
    cv2.circle(dist_transform_copy, (local_max_location[i][1], local_max_location[i][0]), 5, 255)
  # markers = np.zeros_like(dist_transform)
  ret, sure = cv2.threshold(dist_transform, 0.01*dist_transform.max(), 255, 0)
  sure = np.uint8(sure)
  ret, markers = cv2.connectedComponents(sure)
  labels = np.arange(kmeans.n_clusters)
  markers[local_max_location[:,0], local_max_location[:,1]] = labels + 1
  # Convert all local markers to an integer. This because cluster centers will be float numbers. 
  markers = markers.astype(int)
  markers_copy = markers.copy()
  index_non_zero_markers = np.argwhere(markers != 0)
  markers_copy = markers_copy.astype(np.uint8)
  font = cv2.FONT_HERSHEY_SIMPLEX
  for i in range(index_non_zero_markers.shape[0]):
    string_text = str(markers[index_non_zero_markers[i][0], index_non_zero_markers[i][1]])
    cv2.putText(markers_copy, string_text, (index_non_zero_markers[i][1], index_non_zero_markers[i][0]), font, 1, 255)
  markers = markers.astype(np.int32)
  segmented = cv2.watershed(masked, markers)
  count_segments(markers)
  #return {
  #  "count": local_max_location.shape[0]
  #}
  return {
    "count": count_segments(markers),
  }

def count_segments(markers) -> int:
  """
  Count number of unique segments based on value

  keyword arguments:
  markers - numpy array with pixel segmented colour values

  return: int - number of segments
  """
  cnt = Counter()
  for row in markers:
    cnt.update(row)
  n_cnt = dict(takewhile(lambda x: x[1] >= 10, cnt.most_common()))
  del n_cnt[1]
  del n_cnt[-1]
  return len(n_cnt.keys())

def ml_watershed(image, mask) -> JSON_TYPE:
  """
  Apply ML watershedding model to given image to count leaves
  """
  print("Mask: ", mask)
  model_path: str = "saved_models/saved_segmentation_model"
  data_to_predict = tf.data.Dataset.from_tensor_slices(([image], [mask]))
  data_to_predict = data_to_predict.batch(64).prefetch(1)
  model = keras.models.load_model(model_path)
  for im, ma in data_to_predict:
    print("Im: ", im)
    print("Dir im: ", dir(im))
    print(im[0])
    im = cv2.imread((im.numpy()).decode("utf-8"))
    im = cv2.resize(im, (224, 224))
    im = im.reshape(1, 224, 224, 3)
    pred = model.predict(im)
    print("Prediction: ", pred)
  return {
    "count": 0
  }

#!/usr/bin/env python3
"""
  name: evaluate.py
  author: Ryan Jennings
  date: 2021-08-14
"""
import csv
import os

from typing import Dict

from sklearn.metrics import confusion_matrix

from phenotyper.leaf_count import leaf_count

def evaluate() -> None:
  """
  Evaluate PlantCV and OpenCV leaf counting
  algorithms
  """
  images_path: str = "../Leaf_Dataset/images"
  truth_path: str = "../Leaf_Dataset/truth.csv"
  pcv_res = get_leaf_count_results(images_path, "OPENCV")
  print(pcv_res)
  # get_opencv_results(images_path)

  # Analysis
  # Confusion Matrix
  # y_true = []
  # y_pred = []
  # res = confusion_matrix(y_true, y_pred)
  # print("Confusion Matrix: ", res)
  # Precision and Recall
  # F1 Score

class input_file:
  def __init__(self, in_file):
    self.input = in_file

def get_leaf_count_results(images_path: str,
                           model: str) -> Dict[str, int]:
  """
  Run PlantCV or OpenCV leaf counting model across images
  dataset and return predicted results
  """
  files = sorted(os.listdir(f"{images_path}"))
  predictions: Dict[str, int] = {}
  args = {}
  for f in files[:10]:
    input_file_data = input_file(f"{images_path}/{f}")
    leaf_count_res = leaf_count(args=input_file_data,
                                model=model)
    predictions[f] = leaf_count_res['observations']['default']['estimated_object_count']['value']
  return predictions
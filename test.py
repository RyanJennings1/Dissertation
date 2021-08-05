#!/usr/bin/env python3
"""
  name: test.py
  author: Ryan Jennings
  date: 2021-08-04
"""
import cv2

def main() -> None:
  """
  main method
  """
  img_path = './img/arabidopsis.jpg'
  window_name = 'image'
  img = cv2.imread(img_path)
  cv2.imshow(window_name, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()

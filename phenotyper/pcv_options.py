#!/usr/bin/env python3
"""
  name: pcv_options.py
  author: Ryan Jennings
  date: 2021-08-12
"""

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
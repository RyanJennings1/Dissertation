#!/usr/bin/env python3
"""
  name: main.py
  author: Ryan Jennings
  date: 2021-08-05
"""
import json

from typing import Dict, Union

from phenotyper.colour_count import colour_count
from phenotyper.evaluate import evaluate
from phenotyper.leaf_count import leaf_count

def main(args: Dict[str, Union[bool, str]]) -> None:
  """
  main method
  """
  phenotype_data = {
    "leaf_count": None,
    "colour_counts": {}
  }
  print(f'Loaded {args.input}')
  #c_counts = colour_count(args=args)
  #print(c_counts)
  #leaf_count_res = leaf_count(args=args, model="OPENCV")
  #print(json.dumps(leaf_count_res, indent=2))
  #phenotype_data["leaf_count"] = \
  #  leaf_count_res['observations']['default']['estimated_object_count']['value']
  #phenotype_data["colour_counts"] = c_counts
  evaluate()

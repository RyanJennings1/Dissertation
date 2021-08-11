#!/usr/bin/env python3
"""
  name: main.py
  author: Ryan Jennings
  date: 2021-08-05
"""
import json

from typing import Dict, Union

from phenotyper.leaf_count import leaf_count

def main(args: Dict[str, Union[bool, str]]) -> None:
  """
  main method
  """
  phenotype_data = {
    "leaf_count": None
  }
  print(f'Loaded {args.input}')
  leaf_count_res = leaf_count(args=args, method="PLANTCV")
  print(json.dumps(leaf_count_res, indent=2))
  phenotype_data["leaf_count"] = \
    leaf_count_res['observations']['default']['estimated_object_count']['value']

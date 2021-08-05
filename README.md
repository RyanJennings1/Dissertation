# Introduction
This is the repo for my **Plant Phenotyping** MSc. Computer Science Dissertation.  

# :ledger: Index
- [Introduction](#introduction)
- [:ledger: Index](#ledger-index)
- [:beginner: About](#beginner-about)
- [:electric_plug: Installation](#electricplug-installation)
- [:sunny: Usage](#sunny-usage)

# :beginner: About
This repository contains the relevant data for my MSc. Dissertation.  
The dissertation centres around a usable pipeline that is able to identify phenotype traits of a plant given an image such as leaf count, size, colour, and possible diseases.
Plant phenotyping is commonly used in modern agriculture to identify the best characteristics of a plant in order to better breed and produce superior progeny. It is of critical importance to producing increased yields. Key traits of a plant that can be useful to target include fruit size, yield, leaf size, stem thickness, growth rate, colour etc.
For decades phenotyping has been done manually by specially trained cultivists. Major efficiency and yield boosts could be achieved by the elimination of the human factor in phenotyping and with the growth of technology into every sector this seems only more likely.
The intersection of the computer age and agricultural practices has led to some exciting results. While computer vision and machine learning are relatively novel in modern farming they have a vast potential to dramatically improve current processes. Imagine the culmination of years of research and labour being put into a product that is able to accurately identify and remove weeds from a field, that can recognise fruit yields and select individual plants with favourable phenotypes and mutations, and even automatically flagging potentially diseased crops before taking autonomous action to combat further spread.


#  :electric_plug: Installation
```bash
$ git clone https://github.com/RyanJennings1/Dissertation.git
```

Install the project before use to include all the dependencies:
- plantcv

```bash
$ python3 setup.py install
```  
# :sunny: Usage
After installation and being built just run the binary
```bash
$ ./bin/phenotyper --help
usage: phenotyper [-h] [--version]

optional arguments:
  -h, --help     show this help message and exit
  --version, -v  Print version
```

# SARNet
# [Seeing through a Black Box: Toward High-Quality Terahertz Imaging via Subspace-and-Attention Guided Restoration](https://arxiv.org/pdf/2103.16932.pdf)
The source code for the paper: W.-T. Su, Y.-C. Huang, P.-J. Yu, S.-H. Yang, C.-W. Lin “Seeing through a Black Box: Toward High-Quality Terahertz Imaging via Subspace-and-Attention Guided Restoration" in European conference on computer vision, 2022 (ECCV2022) [(Paper link)](https://arxiv.org/pdf/2103.16932.pdf).

## Quick Start
### Installation
**1.Install dependency**
pip install -r requirement.txt

**2.Download the THz Dataset as below:**
https://github.com/wtnthu/THz_data

  *Please put the decompressed data in ./New_data_align

### Run the code
All hyper-parameters can be modified in config.py and run.sh such as:

  *item=deer, channel=64, sub=16

**1.Run run.sh to train the SARNet model. when 2080-Ti GPU, batchsize is 32.**

  *The result world save in the ./output_result

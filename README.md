## DistgDisp: Disentangling Mechanism for Light Field Disparity Estimation
<br>
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgDisp/master/Figs/DistgDisp.png" width="90%"> </p>

***This is the PyTorch implementation of the disparity estimation method in our paper "Disentangling Light Fields for Super-Resolution and Disparity Estimation". Please refer to our [paper](https://yingqianwang.github.io/) and [project page](https://yingqianwang.github.io/DistgLF) for details.***<br>

## Preparation:
### Requirement:
* PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.6, cuda=9.0.
* An AMD3900X CPU and an RTX 3090 GPU are required to reproduce the inference time reported in our paper.

### Datasets:
* We used the HCI 4D LF benchmark for training and evaluation. Please refer to the [benchmark website](https://lightfield-analysis.uni-konstanz.de) for details.

### Path structure:
  ```
  ├──./datasets/
  │    ├── training
  │    │    ├── antinous
  │    │    │    ├── gt_disp_lowres.pfm
  │    │    │    ├── valid_mask.png
  │    │    │    ├── input_Cam000.png
  │    │    │    ├── input_Cam001.png
  │    │    │    ├── ...
  │    │    ├── boardgames
  │    │    ├── ...
  │    ├── validation
  │    │    ├── backgammon
  │    │    │    ├── gt_disp_lowres.pfm
  │    │    │    ├── input_Cam000.png
  │    │    │    ├── input_Cam001.png  
  │    │    │    ├── ...
  │    │    ├── boxes
  │    |    ├── ...
  │    ├── test
  │    │    ├── bedroom
  │    │    │    ├── input_Cam000.png
  │    │    │    ├── input_Cam001.png  
  │    │    │    ├── ...
  │    │    ├── bicycle
  │    |    ├── herbs
  │    |    ├── origami
  ```

## Train:
* Set the hyper-parameters in `parse_args()` if needed. We have provided our default settings in the realeased codes.
* Run `train.py` to perform network training.
* Checkpoint will be saved to `./log/`.

## Test on your own LFs:
* Place the input LFs into `./demo_input` (see the attached examples).
* Run `test.py` to perform inference on each test scene.
* The result files (i.e., `scene_name.pfm`) will be saved to `./Results/`.

## Reproduce the scores on the HCI 4D LF benchmark:
* Perform inference on each scene separately to generate `.pfm` result files.
* Download groundtruth disparity images (i.e., `gt_disp_lowres.pfm`) and use the [official evaluation toolkit](https://github.com/lightfield-analysis/evaluation-toolkit) to obtain quantitative results.


## Results:

### Quantitative Results:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgDisp/master/Figs/QuantitativeDisp.png" width="95%"> </p>

### Visual Comparisons:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgDisp/master/Figs/Visual-Disp.png" width="95%"> </p>

### Screenshot on the HCI 4D LF Benchmark (July 2020):
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgDisp/master/Figs/screenshot.png" width="95%"> </p>

### Performance on real LFs and extended applications:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgDisp/master/Figs/DispApplication.png" width="95%"> </p>


## Citiation
**If you find this work helpful, please consider citing:**
```
@Article{DistgLF,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Disentangling Light Fields for Super-Resolution and Disparity Estimation},
    journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)}, 
    year      = {2022},   
}
```
<br>

## Contact
**Welcome to raise issues or email to [yingqian.wang@outlook.com](yingqian.wang@outlook.com) for any question regarding this work.**

## OACC-Net: Occlusion-Aware Cost Constructor Network
<br>
<p align="center"> <img src="https://raw.github.com/YingqianWang/OACC-Net/master/Figs/OACC-Net.png" width="90%"> </p>

***PyTorch implementation of our paper "Occlusion-Aware Cost Constructor for Light Field Depth Estimation". [[pdf]()]***<br>

## News and Updates:
* 2022-02-28: Codes and models are uploaded.

## Preparation:
### Requirement:
* PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.6, cuda=9.0.
* A single GPU with cuda memory larger than 12 GB is required to reproduce the inference time reported in our paper.

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

## Reproduce the inference time reported in our paper:
* Run `test_inference_time.py` to reproduce the inference time reported in our paper. Note that, the inference need to be performed on a GPU with a cuda memory larger than 12 GB.

## Results:

### Quantitative Results:
<p align="center"> <img src="https://raw.github.com/YingqianWang/OACC-Net/master/Figs/QuantitativeMSE.png" width="95%"> </p>

### Visual Comparisons:
<p align="center"> <img src="https://raw.github.com/YingqianWang/OACC-Net/master/Figs/Visual.png" width="95%"> </p>

### Screenshot on the HCI 4D LF Benchmark (March 2022):
<p align="center"> <img src="https://raw.github.com/YingqianWang/OACC-Net/master/Figs/screenshot.png" width="95%"> </p>

### Performance on real LFs:
<p align="center"> <img src="https://raw.github.com/YingqianWang/OACC-Net/master/Figs/VisualReal.png" width="95%"> </p>

Please refer to our [supplemental material]() for additional quantitative and visual comparisons.

## Citiation
**If you find this work helpful, please consider citing:**
```
@inproceedings{OACC-Net,
  title     = {Occlusion-Aware Cost Constructor for Light Field Depth Estimation},
  author    = {Wang, Yingqian and Wang, Longguang and Liang, Zhengyu and Yang, Jungang and An, Wei and Guo, Yulan},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
<br>

## Contact
**Welcome to raise issues or email to [wangyingqian16@nudt.edu.cn](wangyingqian16@nudt.edu.cn) for any question regarding this work.**

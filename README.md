## OACC-Net
<br>
<p align="center"> <img src="https://raw.github.com/YingqianWang/OACC-Net/master/Figs/OACC-Net.png" width="90%"> </p>

***PyTorch implementation of our paper "Occlusion-Aware Cost Constructor for Light Field Depth Estimation". [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Occlusion-Aware_Cost_Constructor_for_Light_Field_Depth_Estimation_CVPR_2022_paper.pdf)]***<br>

## News and Updates:
* 2022-07-02: Correct a mistake in `train.py`, i.e., dispGT should be fed to the network during training.
* 2022-03-02: Our paper is accepted to CVPR 2022.
* 2022-02-28: Codes and models are uploaded.

## Preparation:
### Requirement:
* PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.
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
<p align="center"> <img src="https://raw.github.com/YingqianWang/OACC-Net/master/Figs/Screenshot.png" width="75%"> </p>

### Performance on real LFs:
<p align="center"> <img src="https://raw.github.com/YingqianWang/OACC-Net/master/Figs/VisualReal.png" width="65%"> </p>

Please refer to our [supplemental material](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wang_Occlusion-Aware_Cost_Constructor_CVPR_2022_supplemental.pdf) for additional quantitative and visual comparisons.

## Citiation
**If you find this work helpful, please consider citing:**
```
@InProceedings{OACC-Net,
    author    = {Wang, Yingqian and Wang, Longguang and Liang, Zhengyu and Yang, Jungang and An, Wei and Guo, Yulan},
    title     = {Occlusion-Aware Cost Constructor for Light Field Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {19809-19818}
}
```
<br>

## Contact
**Welcome to raise issues or email to [wangyingqian16@nudt.edu.cn](wangyingqian16@nudt.edu.cn) for any question regarding this work.**

<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.glitch.me/badge?page_id=YingqianWang/OACC-Net)

</details>


# CRDFormer

## Experiment result

Results on NYU Depth V2 dataset

|   model   | MPA(%) | mIoU(%) | Params/M | FLOPs/G |                            Weight                            |
| :-------: | :----: | :-----: | :------: | :-----: | :----------------------------------------------------------: |
| CRDFormer | 67.73  |  55.51  |  127.9   |  128.2  | [CRDFormer](链接：https://pan.baidu.com/s/1ycFtd9EqKVoWvVIRz3KZiw?pwd=EERR <br/>提取码：EERR ) |

### Requirements

```
python3
timm
mmsegmentation
mmcv
einops
ml_collections
pytorch==2.0.1+cu118
```

Download the pre-training weight of pvt_v2([PVT/classification at v2 · whai362/PVT · GitHub](https://github.com/whai362/PVT/tree/v2/classification))

## How to use

Modify the configuration in file `get_config.py` and run `train.py` or `eval.py`

## Note

The code is partially based on ACNet([GitHub - anheidelonghu/ACNet: ACNet: Attention Complementary Network for RGBD semantic segmentation](https://github.com/anheidelonghu/ACNet)) and mmsegmentation([GitHub - open-mmlab/mmsegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark.](https://github.com/open-mmlab/mmsegmentation)) 


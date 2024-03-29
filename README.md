# AAOD
Adversarial Attack methods on Object Detection <br/>
## Brief
This is an **unfinished** repo for **Adversarial Attack on Object Detection** research based on mmdet.<br/>
We implement some classic attack method such as fgsm, dag and so on.<br/>
[代码数量统计](./code_counter.md)


## Train
单机单卡训练
```bash
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
```
单机多卡训练，需要注意，单机多卡训练时LR由于batch_size的改动需要进行相应的缩放。
```bash
bash .tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --auto-scale-lr \
    [optional arguments]
```
可选的参数包括：
* `--no-validate`：在训练期间关闭测试
* `--work-dir ${WORK_DIR}`：覆盖工作目录
* `--resume-from ${CHECKPOINT_FILE}`：从某个ckpt文件继续训练
* `options 'Key=value'`：覆盖使用的配置文件中的其他设置

**注意：** `resume-from` 和 `load-from`的区别：
`resume-from` 既加载了模型的权重和优化器的状态，也会集成指定ckpt的地带次数，不会重新开始训练。 `load-from`则是只加载魔性的权重，它的训练时从头开始的，经常被用于微调模型。

## Test
单机单卡推理
```bash
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [optional arguments]
```

单机多卡推理
```bash
bash tools/dist_test.sh \
configs/faster_rcnn_r101_fpn_coco.py \
pretrained/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth \
2 \
--work-dir test_results \
--out attack_03_03.pkl
```

## Test Adv
对对抗样本进行推理
```bash
bash test_adv.sh \
    $[MODEL] \ # can be [FR_R101, FR_VGG16, SSD300, CenterNet, DINO]
    $[DATASET] # can be [COCO, VOC]
```

# Citation
以上说明来自于MMDetection官方说明文档。<br/>
本代码库基于OpenMMLab的MMDetection编写，仅用于学术、学习用途。感谢OpenMMLab开发的深度学习框架。
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection/tree/main)
_base_ = ['./base/models/fr_vgg16.py', './base/datasets/coco_datasets.py', './base/default_runtime.py', './base/schedules/schedule_1x.py']

train_dataloader = dict(
    batch_size=4)
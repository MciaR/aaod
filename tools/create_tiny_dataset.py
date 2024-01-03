import json
import os
import pandas as pd
import shutil
from tqdm import tqdm

dataset_type = ['train', 'val']

def create_tiny_coco_dataset(dataset_type, percentage: float = 0.1):  
    annotations_file = 'data/coco2017/annotations/instances_' + dataset_type + '2017.json'
    images_path = 'data/coco2017/images/' + dataset_type + '2017'
    save_dir = 'data/coco2017/annotations/tiny_' + dataset_type + '2017.json'
    target_path = 'data/tiny_coco2017/images/' + dataset_type + '2017'

    with open(annotations_file, "r") as f:
        anno = json.load(f)
    
    pd_images = pd.DataFrame(anno["images"])
    pd_annos = pd.DataFrame(anno["annotations"])
    new_dataset_len = int(len(pd_images) * percentage)
    new_images = pd_images[:new_dataset_len]
    new_images_id = new_images["id"]
    new_annos = pd_annos[pd_annos.image_id.isin(new_images_id)]

    new_images_json = new_images.to_json(orient='records', force_ascii=False)  # to_json将DataFrame转换成了json格式的str
    new_annos_json = new_annos.to_json(orient='records', force_ascii=False)  # loads将str转换为JSON对象
    anno["images"] = json.loads(new_images_json)
    anno["annotations"] = json.loads(new_annos_json)

    anno_json = json.dumps(anno, ensure_ascii=False)  # dumps将JSON对象转换为文件写入支持的类型str

    with open(save_dir, "w") as f:
        f.write(anno_json)

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for img_name in new_images.file_name:
        _source_path = os.path.join(images_path, img_name)
        _target_path = os.path.join(target_path, img_name)
        shutil.copyfile(_source_path, _target_path)

    print(f'Create tiny {dataset_type} dataset successfully!')


def create_tiny_voc_val_dataset(percentage: int = 0.1):
    anno_file = 'data/VOCdevkit/VOC2007_test/ImageSets/Main/test.txt'
    source_root = 'data/VOCdevkit/VOC2007_test/JPEGImages'

    with open(anno_file, 'r') as f:
        anno = f.read().splitlines()

    tiny_ds_length = int(percentage * len(anno))
    tiny_anno = anno[:tiny_ds_length]

    image_save_dir = 'data/VOCdevkit/tiny_voc/JPEGImages'
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    anno_save_dir = 'data/VOCdevkit/tiny_voc/ImageSets/Main/'
    if not os.path.exists(anno_save_dir):
        os.makedirs(anno_save_dir)
    with open(os.path.join(anno_save_dir, 'test.txt'), 'w') as f:
        for line in tiny_anno:
            f.write(line + '\n')

    for img_prefix in tqdm(tiny_anno):
        img_name = img_prefix + '.jpg'
        _source_path = os.path.join(source_root, img_name)
        _target_path = os.path.join(image_save_dir, img_name)
        shutil.copyfile(_source_path, _target_path)

    print(f'Create tiny voc val dataset successfully!')
    


if __name__ == "__main__":

    # create_tiny_coco_dataset(dataset_type=dataset_type[0])
    # create_tiny_coco_dataset(dataset_type=dataset_type[1])
    create_tiny_voc_val_dataset(0.2)
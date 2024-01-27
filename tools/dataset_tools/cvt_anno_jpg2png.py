import json
import os
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

def cvt_anno_jpg2png_coco(anno_path, anno_file_name):
    anno_file = os.path.join(anno_path, anno_file_name)
    with open(anno_file, "r") as f:
        anno = json.load(f)
    
    pd_images = pd.DataFrame(anno["images"])
    pd_images['file_name'] = pd_images['file_name'].str.replace('.jpg', '.png')
    new_images_json = pd_images.to_json(orient='records', force_ascii=False)  # to_json将DataFrame转换成了json格式的str
    anno["images"] = json.loads(new_images_json)
    anno_json = json.dumps(anno, ensure_ascii=False)  # dumps将JSON对象转换为文件写入支持的类型str

    save_dir = os.path.join(anno_path, 'png_' + anno_file_name)
    with open(save_dir, "w") as f:
        f.write(anno_json)

def cvt_anno_jpg2png_voc(anno_path, anno_target_path):
    if not os.path.exists(anno_target_path):
        os.makedirs(anno_target_path)

    for filename in tqdm(os.listdir(anno_path)):
        if filename.endswith('.xml'):
            # 构建完整的文件路径
            filepath = os.path.join(anno_path, filename)

            # 解析 XML 文件
            tree = ET.parse(filepath)
            root = tree.getroot()

            # 查找并更新 filename 标签
            filename_tag = root.find('filename')
            if filename_tag is not None:
                filename_without_ext = os.path.splitext(filename_tag.text)[0]
                filename_tag.text = filename_without_ext + '.png'

            # 保存修改后的 XML 文件
            target_file_path = os.path.join(anno_target_path, filename)
            tree.write(target_file_path)
    
    print("Convert has done!")
    

if __name__ == '__main__':
    anno_path = 'data/coco2017/annotations'
    anno_file_name = 'instances_val2017.json'
    cvt_anno_jpg2png_coco(anno_path, anno_file_name)

    # anno_path = 'data/VOCdevkit/VOC2007_test/Annotations'
    # anno_target_path = 'data/VOCdevkit/VOC2007_test/Annotations_png'
    # cvt_anno_jpg2png_voc(anno_path, anno_target_path)
    
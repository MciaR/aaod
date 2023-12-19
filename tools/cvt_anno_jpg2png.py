import json
import os
import pandas as pd

def cvt_anno_jpg2png(anno_path, anno_file_name):
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


if __name__ == '__main__':
    anno_path = 'data/coco2017/annotations'
    anno_file_name = 'tiny_val2017.json'
    cvt_anno_jpg2png(anno_path, anno_file_name)
    
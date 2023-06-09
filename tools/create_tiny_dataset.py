import json
import pandas as pd

dataset_type = ['train', 'val']

def create_tiny_dataset(dataset_type, percentage: float = 0.1):  
    annotations_file = 'data/coco2014/annotations/instances_' + dataset_type + '2014.json'
    images_path = 'data/coco2014/images/' + dataset_type + '2014'
    save_dir = 'data/coco2014/annotations/tiny_' + dataset_type + '2014.json'

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
    print(f'Create tiny {dataset_type} dataset successfully!')

create_tiny_dataset(dataset_type=dataset_type[0])
create_tiny_dataset(dataset_type=dataset_type[1])
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

    new_images_json = new_images.to_json(orient='records')
    new_annos_json = new_annos.to_json(orient='records')
    anno["images"] = new_images_json
    anno["annotations"] = new_annos_json

    with open(save_dir, "w") as f:
        json.dump(anno, f)
    print(f'Create tiny {dataset_type} dataset successfully!')

create_tiny_dataset(dataset_type=dataset_type[0])
create_tiny_dataset(dataset_type=dataset_type[1])
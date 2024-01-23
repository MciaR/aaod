import json
import os
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

def analysis_dataset_info(dataset_name, annotations_file):
    """Analysis dataset information, including average, max, min amount of classes and objects, average, max, min areas of object and per image, results will be saved as a pic.
    Args:
        dataset_name (str): name of dataset, can be `COCO` or `VOC`.
        annotations_file (str): path of dataset annotation file, must be coco format.
    """

    with open(annotations_file, 'r') as f:
        annos_json = json.load(f)

    images = pd.DataFrame(annos_json['images'])
    annos = pd.DataFrame(annos_json['annotations'])

    classes = []
    object_areas = []
    objects = []

    for img_id in images.id:
        img_annos = annos[annos.image_id == img_id]
        img_classes_list = img_annos['category_id']
        objects.append(len(img_classes_list))
        classes.append(len(set(img_classes_list)))
        object_areas.extend(list(img_annos['area']))
    
    object_areas = list(map(int, object_areas))
    
    row, col = (1, 3)
    plt.figure(frameon=False, figsize=(6*col, 4*row), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    titles = ['Classes Per Image', 'Objects Per Image', 'Area of Objects']
    y_axies = [classes, objects, object_areas]
    for i in range(col):
        plt.subplot(row, col, i + 1)
        if i == 0:
            plt.ylabel(dataset_name, fontsize=16)
        y = y_axies[i]
        x = [ind for ind in range(len(y))]
        y_max = max(y)
        y_min = min(y)
        y_mean = sum(y) / len(y)

        plt.title(f'{titles[i]}')
        plt.axhline(y=y_max, color='#00BCD4', linestyle='-', label=f'Max: {y_max: .2f}')
        # plt.axhline(y=y_min, color='#EC407A', linestyle='-', label=f'Min: {y_min: .2f}')
        plt.axhline(y=y_mean, color='#BA68C8', linestyle='-', label=f'Mean: {y_mean: .2f}')

        plt.scatter(x, y, s=5, color='#FFB300')
        # plt.bar(x, y, color='#FFB300')

        plt.legend(loc='upper right', fontsize=14)
    
    save_dir = f'records/analysis/dataset_info'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, dataset_name + '.png')
    plt.savefig(save_path)


if __name__ == "__main__":
    analysis_dataset_info('COCO', 'data/COCO2017/annotations/instances_val2017.json')
    analysis_dataset_info('VOC', 'data/VOCdevkit/coco_format_annotations/voc07_test.json')
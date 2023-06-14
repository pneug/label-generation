import os
import json

class LabelWriter:
    def __init__(self):
        self.generate_annotation_dict()

    def generate_annotation_dict(self):
        self.annotation_dict = {}
        # label keys: dict_keys(['licenses', 'info', 'categories', 'images', 'annotations'])
        # licenses:  [{'name': '', 'id': 0, 'url': ''}]
        # info:  {'contributor': '', 'date_created': '', 'description': '', 'url': '', 'version': '', 'year': ''}
        # categories:  [{'id': 1, 'name': 'positive', 'supercategory': ''}]
        # sample image:  [{'id': 1, 'width': 1280, 'height': 720, 'file_name': 'bus_data/sunny_2021_03_23_14_33_cam5_filtered/camera5_1616539994_929496759.jpg', 'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}]
        # sample label:  [{'id': 1563, 'image_id': 7974, 'category_id': 1, 'segmentation': [], 'area': 1327.742999999998, 'bbox': [928.24, 295.61, 29.31, 45.3], 'iscrowd': 0, 'attributes': {'occluded': False}}]

        # initialize the annotation dict for coco format
        self.annotation_dict["licenses"] = [{"name": "", "id": 0, "url": ""}]
        self.annotation_dict["info"] = {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""}
        self.annotation_dict["categories"] = [{"id": 1, "name": "positive", "supercategory": ""}]
        self.annotation_dict["images"] = []
        self.annotation_dict["annotations"] = []

    def add_image_entry(self, img_name, img_width=1280, img_height=720):
        # search for the first available id
        img_id = 0
        for img in self.annotation_dict["images"]:
            if img["id"] >= img_id:
                img_id = img["id"] + 1
                
        image_entry = {"id": img_id, "width": img_width,
            "height": img_height, "file_name": img_name, "license": 0, "flickr_url": "",
            "coco_url": "", "date_captured": 0}
        self.annotation_dict["images"].append(image_entry)
        return image_entry["id"]

    def add_annotation_entry(self, img_id, new_box, src_ids, shared_features):
        shared_features_list = []
        for feature in shared_features:
            shared_features_list.append(list(feature))

        # find the first available id
        annotation_id = 0
        for annotation in self.annotation_dict["annotations"]:
            if annotation["id"] >= annotation_id:
                annotation_id = annotation["id"] + 1
        
        annotation_entry = {"id": annotation_id, "image_id": img_id, "category_id": 1, "segmentation": [],
            "area": new_box[2] * new_box[3], "bbox": new_box, "iscrowd": 0, "attributes": {"occluded": False},
            "src_ids": src_ids, "shared_features": shared_features_list}
        print("shared featuresss: ", shared_features_list)
        self.annotation_dict["annotations"].append(annotation_entry)

    def update_annotations(self, reduced_box_dict):
        self.annotation_dict['annotations'] = []
        for img_id, box_groups in reduced_box_dict.items():
            for box_group in box_groups:
                print("box group: ", [0])
                self.add_annotation_entry(img_id, box_group['running_median_box'], 
                box_group['src_ids'], box_group['all_boxes'])
                # self.add_annotation_entry(img_id, box_group['best_box'], 
                # box_group['src_ids'], box_group['shared_features'])

    def get_img_id(self, img_name):
        img_id = None
        for img in self.annotation_dict["images"]:
            if img["file_name"] == img_name:
                img_id = img["id"]
                break
        return img_id

    def save_as_json(self, output_folder, output_name):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # print("annot", self.annotation_dict)

        with open(os.path.join(output_folder, output_name), 'w') as f:
            json.dump(self.annotation_dict, f)
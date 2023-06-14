import numpy as np
import json

class LabelReader:
    def __init__(self, label_path, debug=False):
        with open(label_path) as f:
            self.labels = json.load(f)

        if debug:
            # print all the keys in the labels:
            print("label keys: ", self.labels.keys())
            print("licenses: ", self.labels["licenses"])
            print("info: ", self.labels["info"])
            print("categories: ", self.labels["categories"])
            print("num images: ", len(self.labels["images"]))
            print("num annotations: ", len(self.labels["annotations"]))
            print("sample image: ", self.labels["images"][:1])
            print("sample label: ", self.labels["annotations"][-1:])

    def get_image_with_labels_ids(self):
        # get all image ids for which there are labels
        image_with_labels_ids = []
        for annotation in self.labels["annotations"]:
            if annotation["image_id"] not in image_with_labels_ids:
                image_with_labels_ids.append(annotation["image_id"])
        return image_with_labels_ids

    def get_image_with_labels_names(self, image_with_labels_ids):
        # create a list with all the names and a list with all the gps locations of the images for which there exist labels
        image_with_labels_names = []
        for image in self.labels["images"]:
            if image["id"] in image_with_labels_ids:
                img_name = image["file_name"].split("/")[-1]

                img_name = self.old_to_new_name(img_name)
                image_with_labels_names.append(img_name)
        return image_with_labels_names
    
    # def get_annotations_per_image(self, image_with_labels_ids):
    #     # create a dictionary with all the annotations for each image
    #     annotations_per_image = {}
    #     for image_id in image_with_labels_ids:
    #         annotations_per_image[image_id] = []
    #     for annotation in self.labels["annotations"]:
    #         if annotation["image_id"] in image_with_labels_ids:
    #             annotations_per_image[annotation["image_id"]].append(annotation)
    #     return annotations_per_image
    
    def get_annotations_per_image(self):
        # create a dictionary with all the annotations for each image
        annotations_per_image = {}
        for annotation in self.labels["annotations"]:
            if annotation["image_id"] not in annotations_per_image:
                annotations_per_image[annotation["image_id"]] = []
            annotations_per_image[annotation["image_id"]].append(annotation)
        return annotations_per_image

    def old_to_new_name(self, old_name): 
        # replace the '_' with a '/' to keep cameras in different folders
        return old_name[:7] + '/' + old_name[7 + 1:]

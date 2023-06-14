import json
import os

def merge_coco(label_path, output_path):
    # label_path = "F:\data/label_reconstructions35/generated_labels_single/"
    label_files = os.listdir(label_path)

    merged_labels = None

    for label_file in label_files:
        if merged_labels is None:
            offset = 0
            annotation_offset = 0
        old_to_new_img_id = {}

        # else:
        #     offset = len(merged_labels["images"])
        #     annotation_offset = len(merged_labels["annotations"])
        
        with open(label_path + label_file) as f:
            labels = json.load(f)

        if merged_labels is None:
            merged_labels = labels
            continue

        # old_to_new_img_id = {}
        for i in range(len(labels["images"])):
            old_to_new_img_id[labels["images"][i]["id"]] = offset
            labels["images"][i]["id"] = offset
            offset += 1
            # old_to_new_img_id[labels["images"][i]["id"]] = labels["images"][i]["id"] + offset
        merged_labels["images"] += labels["images"]
        for i in range(len(labels["annotations"])):
            labels["annotations"][i]["image_id"] = old_to_new_img_id[labels["annotations"][i]["image_id"]]
            labels["annotations"][i]["id"] = annotation_offset
            annotation_offset += 1
        merged_labels["annotations"] += labels["annotations"]

    # for label_file in label_files:
    #     if merged_labels is None:
    #         offset = 0
    #         annotation_offset = 0
    #     else:
    #         offset = len(merged_labels["images"])
    #         annotation_offset = len(merged_labels["annotations"])
        
    #     with open(label_path + label_file) as f:
    #         labels = json.load(f)

    #     if merged_labels is None:
    #         merged_labels = labels
    #         continue

    #     # old_to_new_img_id = {}
    #     for i in range(len(labels["images"])):
    #         labels["images"][i]["id"] += offset
    #         # old_to_new_img_id[labels["images"][i]["id"]] = labels["images"][i]["id"] + offset
    #     merged_labels["images"] += labels["images"]
    #     for i in range(len(labels["annotations"])):
    #         labels["annotations"][i]["image_id"] += offset
    #         labels["annotations"][i]["id"] += annotation_offset
    #     merged_labels["annotations"] += labels["annotations"]

    output_file = f"{output_path}/merged_labels.json"
    # rename the file if it already exists
    if os.path.exists(output_file):
        i = 1
        while os.path.exists(output_file):
            output_file = f"{output_path}/merged_labels_{i}.json"
            i += 1
    # create the file
    with open(output_file, "w") as f:
        json.dump(merged_labels, f)

    print("Saved merged labels at ", output_file)

    return merged_labels


if __name__ == "__main__":
    label_path = "F:\data/label_reconstructions35/generated_labels_single/"
    output_path = "F:\data/label_reconstructions35/"
    annotation_dict = merge_coco(label_path, output_path)
    print("generated merged json file at ", f"{output_path}/merged_labels.json")

    re_identification_name = "2021_03_25_14_04"
    from box_generator import BoxGenerator
    boxGenerator = BoxGenerator("F:\data/", "label_reconstructions35", "", -1, re_identification_name)
    print("Saved images to ", f"{output_path}label_visualizations_single_merged")

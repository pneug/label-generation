import json
import argparse
from os import makedirs, path
import matplotlib.pyplot as plt

def main(label_path, label_name, output_path):
    # read the labels
    with open(label_path + label_name + ".json", "r") as f:
        labels = json.load(f)

    images = labels["images"]
    annotations = labels["annotations"]
    annotations_per_img = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in annotations_per_img:
            annotations_per_img[image_id] = []
        annotations_per_img[image_id].append(annotation)

    img_id_to_img = {}
    for image in images:
        img_id_to_img[image["id"]] = image
    
    # check if the output path exists, if yes delete it
    if path.exists(output_path + label_name):
        print("Deleting the existing output path: ", output_path + label_name)
        import shutil
        shutil.rmtree(output_path + label_name)
    output_path = output_path + label_name + "/"
    makedirs(output_path, exist_ok=True)
    
    for image in annotations_per_img:
        image_path = label_path + img_id_to_img[image]["file_name"]
        # print("image_name: ", img_id_to_img[image]["file_name"])

        sunny_label_error = "bus_data/sunny_2021_03_23_14_33_cam5_filtered"
        if sunny_label_error in image_path:
            image_path = image_path.replace(sunny_label_error, "images")

        cloudy_label_error = "bus_data/cloudy_2021_04_09_16_02_cam5_filtered"
        if cloudy_label_error in image_path:
            image_path = image_path.replace(cloudy_label_error, "images")

            # print("image_path: ", image_path)

        img = plt.imread(image_path)
        fig, axis = plt.subplots(1, figsize=(10,10))
        axis.imshow(img)

        # only show the image, no axis
        plt.axis('off')

        for annotation in annotations_per_img[image]:
            bbox = annotation["bbox"]
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            axis.add_patch(rect)
        # create the output path
        file_name = img_id_to_img[image]["file_name"].split("/")[-1]
        file_name = str(image) + "_" + file_name
        plt.savefig(output_path + file_name, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images and GPS from a rosbag.")
    parser.add_argument(
        "-n", "--label_name", default='sunny_reduced_static', help="The name of the label file")
    parser.add_argument(
        "-p", "--label_path", default='../data/Waste_Bin_Detection_Dataset/sunny_2021_03_23_14_33_cam5_filtered (training dataset images and ground truths)', help="The path to the label file")
    parser.add_argument(
        "-o", "--output_path", default='../outputs', help="The path where to save the outputs")
    args = parser.parse_args()

    if len(args.output_path) > 0 and args.output_path[-1] != "/":
        args.output_path += "/"
    if len(args.label_path) > 0 and args.label_path[-1] != "/":
        args.label_path += "/"

    main(args.label_path, args.label_name, args.output_path)
    
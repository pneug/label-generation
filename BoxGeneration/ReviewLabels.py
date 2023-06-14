# In this script, we want to read in all generated labels and then present them while allowing to reject.

import sys, os
import json
import cv2


def get_image_path(labels, image_id):
    # print("image_id: ", image_id)
    for image in labels["images"]:
        # print("image: ", image)
        if image["id"] == image_id:
            return image["file_name"]

def save_labels(labels, label_path, label_name):
    # save the labels
    output_name = label_path + label_name + ".json"
    with open(output_name, "w") as f:
        json.dump(labels, f)
    
    print("labels saved at : ", output_name)

def main(label_path, label_name, img_base_path):
    # read the labels
    with open(label_path + label_name + ".json", "r") as f:
        labels = json.load(f)

    # create a copy of the labels out of which we remove the labels that are rejected
    labels_copy = labels.copy()

    seen_images = []
    labelled_images = []
    for label in labels["annotations"]:
        img_id = label["image_id"]
        labelled_images.append(img_id)
    labelled_images = list(set(labelled_images))

    # present the labelled images
    label_id = -1
    while True:
        label_id += 1
        if label_id >= len(labelled_images):
            label_id = len(labelled_images) - 1
            # seen_images = []
            print("Completed one loop through all images")

        img_id = labelled_images[label_id]

        # label = labels["annotations"][labelled_images[label_id]]
    # for label in labels["annotations"]:
        # print(label)
        # read the image
        # img_id = label["image_id"]
        image_path = get_image_path(labels, img_id)
        # if image_path in seen_images:
        #     print("Skipping image: ", image_path)
        #     continue
        # seen_images.append(image_path)

        image_path = img_base_path + image_path
        # print("image_path: ", image_path)
        # image_path = label["image_path"]
        image = cv2.imread(image_path)
        key = -1
        # print("label", label)

        # find all boxes with the same image_id
        boxes = [l["bbox"] for l in labels["annotations"] if l["image_id"] == img_id]

        # draw in the bounding boxes
        for box in boxes:
            # print("box: ", box)
            box = [int(b) for b in box]
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 1)

        while key == -1:
            # present the image
            print("img_name: ", image_path, image)
            cv2.imshow("image", image)


            # wait for key press
            key = cv2.waitKey(5000)

        # if key is 'r', then reject the label
        if key == ord('r'):
            labelled_images.remove(img_id)
            label_id -= 1
            # remove all labels from the copy that have the same image_id
            labels_copy["annotations"] = [l for l in labels_copy["annotations"] if l["image_id"] != img_id]
            labels_copy["images"] = [l for l in labels_copy["images"] if l["id"] != img_id]
            # labels_copy["annotations"].remove(label)
            print("image removed: ", img_id)

            # labels_copy.remove(label)
        # if key is 'q', then quit the program without saving
        elif key == ord('q'):
            return
        # if key is 's', then save the labels and quit the program
        elif key == ord('s'):
            break
        elif key == ord('p'):
            label_id -= 2
            if label_id < -1:
                print("Completed backwards loop")
                label_id = -1

    # save the labels
    save_labels(labels_copy, label_path, label_name + "_modified")

    labels_copy["images"] = labels["images"]
    save_labels(labels_copy, label_path, label_name + "_modified_all_imgs")


def print_usage():
    print("Usage: python ReviewLabels.py (<label_path>) (<label_name>), using defaults")


if __name__ == "__main__":
    # label_path = "../data/Waste_Bin_Detection_Dataset/sunny_2021_03_23_14_33_cam5_filtered (training dataset images and ground truths)/"
    # label_path = "../data/Waste_Bin_Detection_Dataset/cloudy_2021_04_09_16_02_cam5_filtered (validation and test dataset images and ground truths)/"
    label_path = "../outputs/new/2021_02_18_06_28/"
    # label_name = "labels_sunny_test_modified"
    # label_name = "sunny_2021_03_23_14_33_cam5_filtered_detection_ground_truth_labels(training dataset)"

    # label_name = "unreduced"

    label_name = "merged_labels"

    img_base_path = "../data/generated_trainings_data/"

    label_name = "unreduced"
    # label_name = "detection_validation_dataset_ground_truth_labels(balanced)"
    # read the label_path from the arguments
    # if len(sys.argv) < 2:
    #     print_usage()
    #     # exit(1)
    # else:
    #     label_path = sys.argv[1]

    # if len(sys.argv) < 3:
    #     pass
    #     # exit(1)
    # else:
    #     label_name = sys.argv[2]

    print("Reviewing labels in: ", label_path, label_name)

    main(label_path, label_name, img_base_path)

    #  read in the labels
    versions = ["", "_modified", "_modified_all_imgs"]
    for version in versions:
        with open(label_path + label_name + version + ".json", "r") as f:
            labels = json.load(f)
        print(version, "labels_annotations: ", len(labels["annotations"]))
        print(version, "labels_images: ", len(labels["images"]))

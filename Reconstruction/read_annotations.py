# load the labels
import json
from exif_extended import get_gps_loc

import argparse

def open_annotations(label_path):
    label_path_training = "Waste_Bin_Reconstruction_Sunny/sunny_2021_03_23_14_33_cam5_filtered_detection_ground_truth_labels(training dataset).json"
    if label_path == "":
        label_path = label_path_training
    # label_path_testing = "Waste_Bin_Reconstruction_Sunny/cloudy_2021_04_09_16_02_cam5_filtered_detection_ground_truth_labels.json"
    # data_path = "."
    # data_name = "Waste_Bin_Reconstruction_Sunny"
    # label_path = data_path + data_name + "/cloudy_2021_04_09_16_02_cam5_filtered_detection_ground_truth_labels.json"
    # label_path = data_path + data_name + "/sunny_2021_03_23_14_33_cam5_filtered_detection_ground_truth_labels(training dataset).json"
    with open(label_path) as f:
        labels = json.load(f)
    return labels

def img_name_to_time(img_name):
    img_name = img_name.split("/")[-1]
    img_name = img_name[:-4]  # remove the .jpg
    img_time_whole_seconds = img_name.split("_")[1]
    img_time_split_seconds = img_name.split("_")[2]
    img_time = img_time_whole_seconds + "." + img_time_split_seconds
    img_time = float(img_time)
    return img_time

def img_id_to_name(labels, img_id):
    return labels["images"][img_id-1]["file_name"]

def get_annotation_groups(label_path):
    labels = open_annotations(label_path)
    print("keys of annotations", labels["annotations"][0].keys())

    # check if all annotations are sorted in regard to their 'image_id'
    for i in range(len(labels["annotations"])-1):
        if labels["annotations"][i]["image_id"] < labels["annotations"][i+1]["image_id"]:
            print("not sorted: ", i, 
                img_id_to_name(labels, labels["annotations"][i]["image_id"]), 
                img_id_to_name(labels, labels["annotations"][i+1]["image_id"]))
            break

    # sort all annotations in regard to their 'file_name'
    labels["annotations"].sort(key=lambda x: img_id_to_name(labels, x["image_id"]))
    # labels["images"].sort(key=lambda x: x["file_name"], reverse=True)

    # check if all annotations are sorted in regard to their 'image_id'
    for i in range(len(labels["annotations"])-1):
        if labels["annotations"][i]["image_id"] < labels["annotations"][i+1]["image_id"]:
            print("Stilllll not sorted: ", i, 
                img_id_to_name(labels, labels["annotations"][i]["image_id"]), 
                img_id_to_name(labels, labels["annotations"][i+1]["image_id"]))
            break

    groups = []
    groupNames = []
    newGroupNames = []
    newGroup = []
    newName = img_id_to_name(labels, labels["annotations"][0]["image_id"])
    newGroup.append(img_name_to_time(newName))
    newGroupNames.append(newName)
    # prev_img_time = img_name_to_time(newGroup[0])
    for i in range(1, len(labels["annotations"])):
        new_annot_name = img_id_to_name(labels, labels["annotations"][i]["image_id"])
        img_time = img_name_to_time(new_annot_name)
        if img_time - newGroup[-1] >= 3:
            groups.append(newGroup)
            newGroup = []
            groupNames.append(newGroupNames)
            newGroupNames = []
        newGroup.append(img_time)
        newGroupNames.append(new_annot_name)
        # gps_loc = get_gps_loc(new_annot_name)
        # if gps_loc is None:
        #     newGroupNames.append(None)
        #     # print("No GPS location for ", new_annot_name)
        # else:
        #     newGroupNames.append(new_annot_name)
        #     print("GPS location for ", new_annot_name, " is ", get_gps_loc(new_annot_name))
        # prev_img_time = img_time
    groups.append(newGroup)
    groupNames.append(newGroupNames)

    # groupLocations = calculate_location_boxes(groupLocations)
    return groups, groupNames

def calculate_location_boxes(groupLocations):
    groupLocationsBoxes = []
    for i, group in enumerate(groupLocations):
        print(i, "len group is ", len(group))
        if len(group) == 0:
            # set the box to invalid coordinates
            groupLocationsBoxes.append([200, 200, 200, 200])
            continue
        # print(i, "len group[0] is ", len(group[0]))
        minLat = group[0][0]
        maxLat = group[0][0]
        minLon = group[0][1]
        maxLon = group[0][1]
        for loc in group:
            if loc[0] < minLat:
                minLat = loc[0]
            if loc[0] > maxLat:
                maxLat = loc[0]
            if loc[1] < minLon:
                minLon = loc[1]
            if loc[1] > maxLon:
                maxLon = loc[1]
        groupLocationsBoxes.append([minLat, maxLat, minLon, maxLon])
    return groupLocationsBoxes

# save the locations in a human readable, simple format
def group_locations_to_file(groupLocationsBoxes, file_name):
    with open(file_name, "w") as f:
        for box in groupLocationsBoxes:
            f.write(f"{box[0]}, {box[1]}, {box[2]}, {box[3]}\n")

def groupNamesToLocations(groupNames, path_to_images):
    groupLocations = []
    for i, group in enumerate(groupNames):
        groupLoc = []
        for img_name in group:
            img_path = correct_img_path(img_name, path_to_images, i)
            # print("img_path: ", img_path)
            loc = get_gps_loc(img_path)
            if loc is not None:
                groupLoc.append(loc)
        groupLocations.append(groupLoc)

    print("groupLocations: ", len(groupLocations))
    for i in range(10):
        print("groupLocations[", i, "]: ", len(groupLocations[i]))

    return calculate_location_boxes(groupLocations)

def correct_img_path(img_path, correct_data_path, sequence_num):
    img_name = img_path.split('/')[-1]
    img_name = img_name[:7] + '/' + img_name[7 + 1:] # replace the '_' with a '/' to keep cameras in different folders
    return f"{correct_data_path}/{sequence_num}/images/{img_name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_label_path = "label_reconstructions35/sunny_2021_03_23_14_33_cam5_filtered_detection_ground_truth_labels(training dataset).json"
    parser.add_argument(
        "-l", "--label_path", type=str, default=default_label_path, help="Path to the labels.json file")
    args = parser.parse_args()

    groups, groupNames = get_annotation_groups(args.label_path)
    groupLocationsBoxes = groupNamesToLocations(groupNames, "label_reconstructions35/sequences")

    # print the image ids for the first 10 groups
    for i in range(10):
        print(groups[i])

    # print the image names for the first 10 groups
    for i in range(10):
        print(len(groupNames[i]))

    # print the locations of the first 10 groups
    for i in range(10):
        print(groupLocationsBoxes[i])

    group_locations_to_file(groupLocationsBoxes, "group_locations.txt")
    

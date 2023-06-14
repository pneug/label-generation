import sys, os

import matplotlib.pyplot as plt
import json

from label_writer import LabelWriter
from ImageAnnotator import ImageAnnotator

def create_result_dict():
    return {
    "Good": 0,
    "Low IoU": 0,
    "Missed some objects": 0,
    "Missed but nearby label exists": 0,
    "False Positive": 0
}

def intersection_over_union(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the area of the two boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Find the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Check if there is an overlap between the boxes
    if x_right > x_left and y_bottom > y_top:
        # Calculate the area of the intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        # Calculate the IoU
        IoU = intersection_area / (area1 + area2 - intersection_area)
    else:
        IoU = 0
    return IoU

class Evaluator:
    def __init__(self, generated_labels_path, run, generated_labels_name=""):
        self.annotation_file_name = "generation_test_annotations.json"
        self.load_or_create_annotations()

        self.generated_labels_path = generated_labels_path
        self.run = run
        self.generated_labels_name = generated_labels_name
        self.load_generated_labels()

        self.excluded_images = ["1642804066_700761922.jpg", "1642794993_026760267.jpg", "1613649527_563673270.jpg", "1613660970_928659396.jpg", "1613651050_729522290.jpg", "1613660972_523641122.jpg", "1613660972_523641122.jpg", "1613651051_330365111.jpg", "1613651051_330365111.jpg",
                                "1642804062_103107946.jpg", "1642794995_026394484.jpg", "1642794995_695279962.jpg", "1642804062_295954483.jpg",
                                "1642794993_895708497.jpg", "1613660972_523641122.jpg", "1613651051_330365111.jpg", "1642804066_097828137.jpg",
                                "1642804062_103107946.jpg"]

    def load_or_create_annotations(self):
        self.label_writer = LabelWriter()

        # read the labels, if they exist
        if os.path.isfile(self.annotation_file_name):
            # label_reader = LabelReader(annotation_file_name)
            # labels = label_reader.labels
            with open(self.annotation_file_name, "r") as f:
                labels = json.load(f)
            self.label_writer.annotation_dict = labels
            print("labels loaded")

    def show_existing_annotations(self):
        if False:
            # Show the labels
            for image in self.label_writer.annotation_dict["images"]:
                img_path = image["file_name"]
                img = plt.imread(img_path)
                fig, axis = plt.subplots(1, figsize=(10,10))
                axis.imshow(img)
                # add the image name as the title
                axis.set_title(img_path.split("/")[-1])
                print("image: ", img_path.split("/")[-1])
                axis.axis('off')
                for label in self.label_writer.annotation_dict["annotations"]:
                    if label["image_id"] == image["id"]:
                        x, y, w, h = label["bbox"]
                        # print("Drawing box: ", x, y, w, h)
                        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                        axis.add_patch(rect)
                plt.show()

    def show_generated_labels(self, save_only=True, save_name=""):
        # delete the output folder if it exists and create a new one
        output_folder = "../outputs/evaluated_labels/" + self.run + "/"
        if save_name != "":
            output_folder += save_name + "/"
        if os.path.exists(output_folder):
            import shutil
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
            
        # Show the labels
        for image in self.generated_labels["images"]:
            img_path = self.generated_labels_path + self.run + image["file_name"]
            img = plt.imread(img_path)
            fig, axis = plt.subplots(1, figsize=(10,10))
            axis.imshow(img)
            # add the image name as the title
            axis.set_title(self.individual_image_results.get(img_path.split("/")[-1], "excluded"))
            axis.axis('off')
            for label in self.generated_labels["annotations"]:
                if label["image_id"] == image["id"]:
                    x, y, w, h = label["bbox"]
                    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                    axis.add_patch(rect)
            # draw in the gt labels if they exist
            for image in self.label_writer.annotation_dict["images"]:
                if image["file_name"].split("/")[-1] == img_path.split("/")[-1]:
                    img_id = image["id"]
                    for label in self.label_writer.annotation_dict["annotations"]:
                        if label["image_id"] == img_id:
                            x, y, w, h = label["bbox"]
                            # print("Drawing box: ", x, y, w, h)
                            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
                            axis.add_patch(rect)
            if not save_only:
                plt.show()

            # save the image
            saved_img_name = img_path.split("/")[-1]
            plt.savefig(output_folder + "jpg_" + saved_img_name, bbox_inches='tight', pad_inches=0.0, dpi=250)
            saved_img_name = saved_img_name.replace(".jpg", ".pdf")
            plt.savefig(output_folder + saved_img_name, bbox_inches='tight', pad_inches=0.0, dpi=250)
            axis.set_title("")
            plt.savefig(output_folder + "raw_" + saved_img_name, bbox_inches='tight', pad_inches=0.0, dpi=250)
            plt.close()

    def load_generated_labels(self):
        if not self.generated_labels_name.endswith(".json"):
            # read out all the files in the generated labels folder. Take the newest file
            path = self.generated_labels_path + self.run
            files = os.listdir(path)
            highest_num = 0
            file_name = "merged_labels.json"
            for i in range(len(files)):
                if "merged_labels_" in files[i]:
                    num = int(files[i].split("_")[-1].split(".")[0])
                    if num > highest_num:
                        highest_num = num
                        file_name = files[i]
            # assert "merged" in files[-1], "The newest file in the generated labels folder does not contain 'merged' in the name"
            # assert "merged" in files[-1]
            file_name = path + file_name
        else:
            file_name = path + self.generated_labels_name

        print("Using label_file ", file_name)
        with open(file_name, "r") as f: # self.generated_labels_path + 
            self.generated_labels = json.load(f)

    def evaluate(self):
        label_results = create_result_dict()
        image_results = create_result_dict()
        individual_image_results = {}
        num_excluded_images = 0
        for image in self.generated_labels["images"]:
            img_name = image["file_name"].split("/")[-1]
            if img_name in self.excluded_images:
                num_excluded_images += 1
                continue
            gt_image_names = [img["file_name"].split("/")[-1] for img in self.label_writer.annotation_dict["images"]]
            if img_name not in gt_image_names:
                print("Creating label for image: ", img_name, image["file_name"])
                print(self.generated_labels_path + self.run + image["file_name"])
                image_annotator = ImageAnnotator()
                full_path = self.generated_labels_path + self.run + image["file_name"]
                continue_labeling = image_annotator.label_image(full_path)
                if not continue_labeling:
                    break

                if image_annotator.exclude_image:
                    self.excluded_images.append(img_name)
                    num_excluded_images += 1
                    continue

                assert image_annotator.image.shape[0] == image["height"], "The labeled image has a different height than the generated image"
                assert image_annotator.image.shape[1] == image["width"], "The labeled image has a different width than the generated image"

                new_img_id = self.label_writer.add_image_entry(full_path, image["width"], image["height"])
                boxes = image_annotator.boxes
                boxes = [[box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]] for box in boxes]
                for box in boxes:
                    self.label_writer.add_annotation_entry(new_img_id, box, [], [])
                self.label_writer.save_as_json(".", self.annotation_file_name)
            else:
                # get all the boxes from the annotations
                gt_img_id = [img["id"] for img in self.label_writer.annotation_dict["images"] if img["file_name"].split("/")[-1] == img_name][0]
                boxes = []
                for label in self.label_writer.annotation_dict["annotations"]:
                    if label["image_id"] == gt_img_id:
                        boxes.append(label["bbox"])

            # get the boxes into the correct format (x, y, w, h) with x and y being the top left corner and w, h > 0
            new_boxes = []
            for box in boxes:
                if box[2] < 0:
                    box[0] += box[2]
                    box[2] *= -1
                if box[3] < 0:
                    box[1] += box[3]
                    box[3] *= -1
                new_boxes.append(box)
            boxes = new_boxes
            
            # get all the boxes from the generated labels
            generated_boxes = []
            for label in self.generated_labels["annotations"]:
                if label["image_id"] == image["id"]:
                    generated_boxes.append(label["bbox"])
            
            # match the boxes based on distance
            # for each box in the generated labels, find the closest box in the gt labels, with a maximum distance of 2x the size of the box
            # if there is no box within x pixels, then the box is a false positive

            list_of_distances = []
            distance_dict = {}
            for i, box in enumerate(generated_boxes):
                distances = []
                for j, gt_box in enumerate(boxes):
                    # compute the distance between the boxes
                    distance = ((box[0] - gt_box[0]) ** 2 + (box[1] - gt_box[1]) ** 2) ** 0.5
                    distance_dict[(i, j)] = distance
                    distances.append(distance)
                list_of_distances.append(distances)
            
            # find the best match for each box
            best_matches = []
            for i in range(len(list_of_distances)):
                best_match = -1
                # set the best distance to 2x the size of the box
                generated_box = generated_boxes[i]
                best_distance = 2 * (generated_box[2] ** 2 + generated_box[3] ** 2) ** 0.5
                # best_distance = 2 * (list_of_distances[i][0][2] ** 2 + list_of_distances[i][0][3] ** 2) ** 0.5
                # best_distance = 10000
                for j in range(len(list_of_distances[i])):
                    if list_of_distances[i][j] < best_distance:
                        best_distance = list_of_distances[i][j]
                        best_match = j
                best_matches.append(best_match)
            
            current_img_state = "Good"
            
            # check if there are any boxes that are matched to the same box
            for i in range(len(best_matches)):
                for j in range(i + 1, len(best_matches)):
                    if best_matches[i] == best_matches[j] and best_matches[i] != -1:
                        # print("image: ", img_name)
                        # print("distances: ", distance_dict[(i, best_matches[i])], distance_dict[(j, best_matches[j])])
                        print("boxes: ", generated_boxes[i], boxes[best_matches[i]])
                        print("Box ", i, " and box ", j, " are matched to the same box")

                        # # check which box is closer, set the other one to -1
                        # if distance_dict[(i, best_matches[i])] < distance_dict[(j, best_matches[j])]:
                        #     best_matches[j] = -1
                        # else:
                        #     best_matches[i] = -1

                        # # draw this image
                        # img_path = self.generated_labels_path + self.run + image["file_name"]
                        # img = plt.imread(img_path)
                        # fig, axis = plt.subplots(1, figsize=(10,10))
                        # axis.imshow(img)
                        # # add the image name as the title
                        # axis.set_title(img_path.split("/")[-1])
                        # print("image: ", img_path.split("/")[-1])
                        # axis.axis('off')
                        # for label in self.generated_labels["annotations"]:
                        #     if label["image_id"] == image["id"]:
                        #         x, y, w, h = label["bbox"]
                        #         # print("Drawing box: ", x, y, w, h)
                        #         rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                        #         axis.add_patch(rect)
                        # for label in self.label_writer.annotation_dict["annotations"]:
                        #     if label["image_id"] == gt_img_id:
                        #         x, y, w, h = label["bbox"]
                        #         # print("Drawing box: ", x, y, w, h)
                        #         rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
                        #         axis.add_patch(rect)
                        # plt.show()

                        # assert False, "Box " + str(i) + " and box " + str(j) + " are matched to the same box"
            
            # save the results for each box and image
            for i in range(len(best_matches)):
                if best_matches[i] == -1:
                    # no match found
                    label_results["False Positive"] += 1
                    current_img_state = "False Positive"
                else:
                    # check if the IoU is high enough
                    box = generated_boxes[i]
                    gt_box = boxes[best_matches[i]]
                    iou = intersection_over_union(box, gt_box)
                    # print("iou: ", iou, box, gt_box)
                    if iou < 0.01:
                        label_results["Missed but nearby label exists"] += 1
                        if current_img_state == "Good" or current_img_state == "Low IoU":
                            current_img_state = "Missed but nearby label exists"
                    elif iou < 0.5:
                        label_results["Low IoU"] += 1
                        if current_img_state == "Good":
                            current_img_state = "Low IoU"
                    else:
                        label_results["Good"] += 1
            
            # check if any boxes are missing
            for i in range(len(boxes)):
                if i not in best_matches:
                    # no match found
                    label_results["Missed some objects"] += 1
                    current_img_state = "Missed some objects"
            
            image_results[current_img_state] += 1
            individual_image_results[img_name] = current_img_state
            

        # self.label_writer.save_as_json(".", self.annotation_file_name)
        print(f"Exluded {num_excluded_images} images from the evaluation".format(num_excluded_images))
        print("label results: ", label_results)
        print("image results: ", image_results)
        self.individual_image_results = individual_image_results
        return label_results, image_results
    
    def save_positive_labels(self):
        # save all labels classified as "Good" in a json file
        generated_labels = self.generated_labels
        cleaned_labels = generated_labels.copy()
        new_annotations = []
        new_images = []
        for label in generated_labels["annotations"]:
            img_name = [img["file_name"].split("/")[-1] for img in generated_labels["images"] if img["id"] == label["image_id"]][0]
            if img_name in self.individual_image_results and self.individual_image_results[img_name] == "Good":
                new_annotations.append(label)
                if label["image_id"] not in [img["id"] for img in new_images]:
                    new_images.append([img for img in generated_labels["images"] if img["id"] == label["image_id"]][0])
        cleaned_labels["annotations"] = new_annotations
        cleaned_labels["images"] = new_images

        with open(self.generated_labels_path + self.run + "cleaned_labels.json", "w") as f:
            json.dump(cleaned_labels, f)
    
    def remove_img(self, img_name):
        if input("Are you sure you want to remove image " + img_name + "? (y/n)") != "y":
            return
        
        # create a backup
        import shutil
        shutil.copyfile(self.annotation_file_name, self.annotation_file_name[:-5] + "_backup.json")
        print(f"Created backup of {self.annotation_file_name} to {self.annotation_file_name[:-5] + '_backup.json'}")
        
        img_id = -1
        # remove the image from the hand-labeled images
        for i in range(len(self.label_writer.annotation_dict["images"])):
            if self.label_writer.annotation_dict["images"][i]["file_name"].split("/")[-1] == img_name:
                img_id = self.label_writer.annotation_dict["images"][i]["id"]
                del self.label_writer.annotation_dict["images"][i]
                print("Successfully removed image: ", img_name)
                break

        if img_id == -1:
            print("Image not found")
            return

        # remove the annotations for the image
        annotations_to_remove = []
        for j in range(len(self.label_writer.annotation_dict["annotations"])):
            if self.label_writer.annotation_dict["annotations"][j]["image_id"] == img_id:
                # del label_writer.annotation_dict["annotations"][j]
                annotations_to_remove.append(j)
                # break

        annotations_to_remove.sort(reverse=True)
        for j in annotations_to_remove:
            del self.label_writer.annotation_dict["annotations"][j]
            print("Successfully removed annotation for image: ", img_name)
        
        # save the labels
        self.label_writer.save_as_json(".", self.annotation_file_name)



if __name__ == "__main__":
    # label_path = sys.argv[1]
    # generated_labels_path = "../outputs/"
# run = "2022_01_21_14_04/"
# generated_labels_name = ""
    evaluator = Evaluator("../outputs/", "2022_01_21_14_04/")
    evaluator.show_existing_annotations()
    # evaluator.evaluate()
    label_results, image_results = evaluator.evaluate()
    evaluator.show_generated_labels()
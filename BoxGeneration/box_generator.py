import numpy as np
import sys, os

from matplotlib import pyplot as plt
import argparse

import shutil
import json

from label_writer import LabelWriter
from label_reader import LabelReader
from box_group_combinator import combine_grouped_boxes

sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), "..", "colmap", "scripts", "python"))
from read_write_model import read_images_binary, read_points3D_binary

# sequences_with_one_label = []

# DRAW_VISUALIZATIONS = True
SAFE_NEW_LABELS = True


class BoxGenerator:
    no_sparse_model_count = 0
    no_features_count = 0
    features_count = 0
    box_generated_count = 0
    no_box_generated_count = 0
    img_generated_count = 0
    no_img_generated_count = 0
    generated_labels_per_sequence = {}

    def __init__(self, data_path, reconstruction_folder_name, label_file_name, sequence_id, re_identification_name, min_shared_features, draw_visualizations, debug=False):

        self.labelWriter = LabelWriter()
        # self.reconstruction_folder = reconstruction_folder
        # db_path = data_path + data_name + sequence + "/database.db"
        self.reconstruction_folder = data_path + reconstruction_folder_name
        self.generated_label_path = self.reconstruction_folder + "/generated_labels/" + re_identification_name + "/"
        # label_file_name = "/sunny_2021_03_23_14_33_cam5_filtered_detection_ground_truth_labels(training dataset).json"
        # label_path = data_path + reconstruction_folder + "/cloudy_2021_04_09_16_02_cam5_filtered_detection_ground_truth_labels.json"
        self.label_path = self.reconstruction_folder + label_file_name
        self.sequence_id = sequence_id
        self.sequence = "/sequences/" + str(self.sequence_id)
        self.images_path = self.reconstruction_folder + self.sequence + "/images/"
        self.re_identification_name = re_identification_name
        self.debug = debug
        self.min_shared_features = min_shared_features
        self.draw_visualizations = draw_visualizations

    def generate_boxes(self, num_labels_per_sequence, use_mean, sampling_scheme):
        path_to_sparse_model = os.path.join(self.images_path, self.re_identification_name, "new-sparse-model")
        if not os.path.exists(path_to_sparse_model):
            print("no sparse model found for sequence " + str(self.sequence_id) + " at path " + path_to_sparse_model)
            BoxGenerator.no_sparse_model_count += 1
            return
        images = read_images_binary(os.path.join(path_to_sparse_model, "images.bin"))
        points3D = read_points3D_binary(os.path.join(path_to_sparse_model, "points3D.bin"))
        if self.debug:
            print("length of points3D", len(points3D))
            print("length of images", len(images))
            print(list(images.keys())[:1])
            print(list(images.values())[:1])
            print(list(points3D.keys())[:1])
            print(list(points3D.values())[:1])

        labelReader = LabelReader(self.label_path)
        img_with_labels_ids = labelReader.get_image_with_labels_ids()
        image_with_labels_names = labelReader.get_image_with_labels_names(img_with_labels_ids)

        if self.debug:
            print("num images with labels: ", len(img_with_labels_ids))

        # from box_generator import get_img_with_labels_ids
        img_with_labels_ids = self.get_img_with_labels_ids_in_sequence(images, image_with_labels_names, img_with_labels_ids)

        self.generate_box_dict(labelReader.labels, img_with_labels_ids)

        box_points_per_image = self.get_box_points_per_image(images, img_with_labels_ids)

        print("box_points_per_image", len(images), box_points_per_image)
        if len(box_points_per_image) > 0:
            BoxGenerator.features_count += 1
        else:
            BoxGenerator.no_features_count += 1
        # return
        #for im, points in box_points_per_image.items():
        #    print("im", im)
        #    print("len(im)", len(points))
        #    if len(points) > 1:
        #        return

        #sequences_with_one_label.append(self.sequence_id)
        
        # print("len(box_points_per_image)", len(box_points_per_image))
        #return

        box_points_in_new_images = self.get_box_points_in_new_images(images, box_points_per_image)

        self.generate_new_annotations(box_points_in_new_images, images, points3D, use_mean)

        # return # TODO!!!

        combine_grouped_boxes(self.labelWriter)

        # print("Num labels, num images", len(self.labelWriter.annotation_dict['annotations']), len(self.labelWriter.annotation_dict['images']))
        if len(self.labelWriter.annotation_dict['annotations']) > 0 and len(self.labelWriter.annotation_dict['images']) > 0:
            BoxGenerator.box_generated_count += 1
        else:
            BoxGenerator.no_box_generated_count += 1

        # src_annotations_per_image = labelReader.get_annotations_per_image()
        # self.reduce_labels_per_sequence(src_annotations_per_image, num_labels_per_sequence, sampling_scheme)
        
        # BoxGenerator.generated_labels_per_sequence[self.sequence_id] = len(self.labelWriter.annotation_dict['annotations'])
        # save how many labeled images were generated for this sequence
        BoxGenerator.generated_labels_per_sequence[self.sequence_id] = len(self.labelWriter.annotation_dict['images'])

        self.labelWriter.save_as_json(self.generated_label_path, f"labels_{self.sequence_id}.json") # f"labels_{self.sequence_id}.json"

    def get_img_with_labels_ids_in_sequence(self, images, image_with_labels_names, image_with_labels_ids):
        # match the img ids from the labels with the img ids from the database based on the image names
        img_with_labels_ids = []
        self.label_img_id_to_db_img_id = {}
        for img_with_labels_name, image_with_labels_id in zip(image_with_labels_names, image_with_labels_ids):
            for img_id, img_data in images.items():
                if img_with_labels_name in img_data.name:
                    img_with_labels_ids.append(img_id)
                    self.label_img_id_to_db_img_id[image_with_labels_id] = img_id
                    break
        
        self.db_img_id_to_label_img_id = {v: k for k, v in self.label_img_id_to_db_img_id.items()}
        return img_with_labels_ids

    def generate_box_dict(self, labels, img_with_labels_ids):
        # extract the boxes from the label annotations into a dictionaries for which the img id is the key
        self.box_dict = {}
        for img_id in img_with_labels_ids:
            img_boxes = []
            for annotation in labels['annotations']:
                if self.label_img_id_to_db_img_id.get(annotation['image_id'], -1) == img_id:
                    img_boxes.append(annotation['bbox'])
            self.box_dict[img_id] = img_boxes

    def get_box_points_per_image(self, images, img_with_labels_ids):
        box_points_per_image = {}
        for img_with_labels_id in img_with_labels_ids:
            box_points_per_image[img_with_labels_id] = {}
            points_2d_in_image = images[img_with_labels_id].xys
            points_in_image = images[img_with_labels_id].point3D_ids
            for point_2d, point in zip(points_2d_in_image, points_in_image):
                # check if the point is in any of the boxes of the image
                for box_id, box in enumerate(self.box_dict[img_with_labels_id]):
                    if box_id not in box_points_per_image[img_with_labels_id]:
                        box_points_per_image[img_with_labels_id][box_id] = []
                    if self.box_contains_point(box, point_2d):
                        box_points_per_image[img_with_labels_id][box_id] = box_points_per_image[img_with_labels_id][box_id] + [point]
        return box_points_per_image

    def box_contains_point(self, box, point):
        return point[0] > box[0] and point[0] < box[0] + box[2] and point[1] > box[1] and point[1] < box[1] + box[3]

    def get_box_points_in_new_images(self, images, box_points_per_image):
        # make a dict that contains for all new images the corresponding src img and points per src img
        box_points_in_new_images = {}
        for img_id, img_data in images.items():
            img_path = img_data.name
            if not self.re_identification_name in img_path:
                continue

            # check if any of the src images contains a point that is contained in this image
            for src_img_id, src_box_data in box_points_per_image.items():
                for box_id, points_3d in src_box_data.items():
                    for point_3d in points_3d:
                        if point_3d in img_data.point3D_ids and point_3d != -1:
                            if img_id not in box_points_in_new_images:
                                box_points_in_new_images[img_id] = {}
                            if src_img_id not in box_points_in_new_images[img_id]:
                                box_points_in_new_images[img_id][src_img_id] = {}
                            if box_id not in box_points_in_new_images[img_id][src_img_id]:
                                box_points_in_new_images[img_id][src_img_id][box_id] = []

                            box_points_in_new_images[img_id][src_img_id][box_id].append(point_3d)

                    if img_id in box_points_in_new_images and src_img_id in box_points_in_new_images[img_id] and box_id in box_points_in_new_images[img_id][src_img_id]:
                        box_points_in_new_images[img_id][src_img_id][box_id] = np.array(box_points_in_new_images[img_id][src_img_id][box_id], dtype=np.int32)
        return box_points_in_new_images

    # ---------------------------- Box Generation ----------------------------

    def generate_new_annotations(self, box_points_in_new_images, images, points3D, use_mean):
        # visualize the new points using plt
        for img_id, box_points in box_points_in_new_images.items():
            plt.close()
            if img_id != 1183:
                continue
            if self.draw_visualizations:
                img_path = images[img_id].name
                img = plt.imread(os.path.join(self.images_path, img_path))
                plt.imshow(img)
                plt.axis('off')

            src_images = list(box_points.keys())
            should_draw = False
            draw_target = True
            src_boxes = []
            target_boxes = []
            print("Num src images", len(src_images))

            for src_img_num, (src_img_id, box_points) in enumerate(box_points_in_new_images[img_id].items()):
                # if draw_visualizations and should_draw:
                #     break # TODO
                for box_id, points_3d_ids in box_points.items():
                    target_points_2d = self.get_2d_points_of_3d_point3(points_3d_ids, images[img_id])
                    src_points_2d = self.get_2d_points_of_3d_point3(points_3d_ids, images[src_img_id])

                    if len(target_points_2d) < self.min_shared_features:
                        continue
                    
                    new_box = self.calculate_new_box(self.box_dict, src_img_id, box_id, images, points3D, img_id, points_3d_ids, target_points_2d, src_points_2d, use_mean)

                    src_boxes.append(self.box_dict[src_img_id][box_id])
                    target_boxes.append(new_box)

                    if not draw_target and self.draw_visualizations and src_img_num == 0:
                        # draw the source image
                        src_img_path = images[src_img_id].name
                        src_img = plt.imread(os.path.join(self.images_path, src_img_path))
                        plt.imshow(src_img)
                        plt.axis('off')

                        # draw in the source label and keypoints
                        src_box = self.box_dict[src_img_id][box_id]
                        plt.scatter(src_points_2d[:, 0], src_points_2d[:, 1], c='r', s=10)
                        # draw in the median point in green
                        median_point_src = np.median(src_points_2d, axis=0)
                        plt.scatter(median_point_src[0], median_point_src[1], c='g', s=20)

                        axis = plt.gca()
                        axis.add_patch(plt.Rectangle((src_box[0], src_box[1]), src_box[2], src_box[3], fill=False, edgecolor='r', lw=2))
                        should_draw = True
                        # break # TODO
                    
                    if draw_target and self.draw_visualizations: #  and src_img_num == 0:
                        source_colors = ['r'] # , 'g', 'b', 'y', 'c', 'm', 'w']
                        col = source_colors[src_img_num % len(source_colors)]
                        # print("col", col)
                        # plt.scatter(target_points_2d[:, 0], target_points_2d[:, 1], c=col, s=10)

                        # draw in the median point in green
                        median_point_target = np.median(target_points_2d, axis=0)
                        # plt.scatter(median_point_target[0], median_point_target[1], c='g', s=30)
                        print("Drawing in box, ", src_img_num, src_img_id)

                        axis = plt.gca()
                        axis.add_patch(plt.Rectangle((new_box[0], new_box[1]), new_box[2], new_box[3], fill=False, edgecolor=col, lw=1))
                        should_draw = True
                        # break # TODO

                        # add a label to the rectangle that states the scaling factor
                        # random_offset = np.random.rand() * 200
                        # axis.text(new_box[0] + 10, new_box[1] + src_img_num * 40, str(round(scale_factor, 3)), color=col)

                    if SAFE_NEW_LABELS:
                        # print("target_points_2d", target_points_2d)
                        self.generate_label_dict(new_box, img_id, src_img_id, target_points_2d, images)

            if draw_visualizations and should_draw:
                if len(src_images) <= 1:
                    plt.close()
                    continue

                # plt.show()
                # save the visualization
                output_path_vis = "../outputs/box_visualizations/"
                os.makedirs(output_path_vis, exist_ok=True)
                # full_fig_path = output_path_vis + f"/target_{img_id}_src_is_{src_images[0]}_{len(src_images)}.pdf"

                if draw_target:
                    vis_box = new_box
                    padding = 200
                    full_fig_path = output_path_vis + f"/target_{img_id}_src_is_{src_images[0]}_{self.sequence_id}.pdf"
                else:
                    vis_box = src_box
                    padding = 300
                    full_fig_path = output_path_vis + f"/src_{src_images[0]}_{len(src_images)}.pdf"

                # zoom in on the box but keep the aspect ratio
                if True:
                    plt.xlim(vis_box[0] - padding, vis_box[0] + vis_box[2] + padding)
                    x_size = vis_box[2] + padding * 2
                    original_aspect_ratio = img.shape[1] / img.shape[0]
                    y_size = x_size / original_aspect_ratio
                    box_y_mid = vis_box[1] + vis_box[3] / 2
                    plt.ylim(box_y_mid + y_size / 2, box_y_mid - y_size / 2)
                
                # plt.savefig(full_fig_path, bbox_inches='tight', pad_inches=0, dpi=200)
                full_fig_path = full_fig_path.replace(".pdf", ".png")
                plt.savefig(full_fig_path, bbox_inches='tight', pad_inches=0, dpi=200)
                # print("saved visualization for", img_path)
                plt.close()

    def calculate_new_box(self, box_dict, src_img_id, box_id, sparse_images, points3D, img_id, points_3d_ids, target_points_2d, src_points_2d, use_mean):
        src_box = box_dict[src_img_id][box_id]

        if use_mean:
            median_point_target = np.mean(target_points_2d, axis=0)
            median_point_src = np.mean(src_points_2d, axis=0)
        else:
            median_point_target = np.median(target_points_2d, axis=0)
            median_point_src = np.median(src_points_2d, axis=0)

        # error = np.mean([points3D[point_3d_id].error for point_3d_id in points_3d_ids])
        # print(f"Error: {error}, using {len(points_3d_ids)} points for img {img_id} and src img {src_img_id}")

        scale_factor = self.calculate_scale_factor(sparse_images, points3D, img_id, src_img_id, points_3d_ids)

        new_box_width = src_box[2] * scale_factor
        new_box_height = src_box[3] * scale_factor

        # the src box is of format [x_min, y_min, w, h]
        src_box_mean = np.array([src_box[0] + src_box[2] / 2, src_box[1] + src_box[3] / 2])
        median_to_center_offset_src = src_box_mean - median_point_src
        scaled_median_to_center_offset_src = median_to_center_offset_src * scale_factor

        new_box = [median_point_target[0] + scaled_median_to_center_offset_src[0] - new_box_width / 2,
            median_point_target[1] + scaled_median_to_center_offset_src[1] - new_box_height / 2,
            new_box_width, new_box_height]
        return new_box

    def generate_label_dict(self, new_box, target_img_id, source_img_id, shared_features, images):
        # check if the image is already in the annotation dict
        img_id = self.labelWriter.get_img_id(images[target_img_id].name)
        
        if img_id is None:
            img_id = self.labelWriter.add_image_entry(images[target_img_id].name)
        
        self.labelWriter.add_annotation_entry(img_id, new_box, [self.db_img_id_to_label_img_id[source_img_id]], shared_features)

    def calculate_scale_factor(self, sparse_images, points3D, img_id, src_img_id, points_3d_ids):
        src_cam_world_pos = self.get_image_world_pos(sparse_images[src_img_id])
        target_cam_world_pos = self.get_image_world_pos(sparse_images[img_id])
        
        points_3d = np.array([points3D[point_3d_id].xyz for point_3d_id in points_3d_ids])
        mean_3d_point = np.mean(points_3d, axis=0)
        points_3d_to_src_dist = np.linalg.norm(mean_3d_point - src_cam_world_pos)
        points_3d_to_target_dist = np.linalg.norm(mean_3d_point - target_cam_world_pos)
        
        scale_factor = points_3d_to_src_dist / points_3d_to_target_dist
        return scale_factor

    def get_image_world_pos(self, image):
        return -np.transpose(image.qvec2rotmat()) @ image.tvec

    def get_2d_points_of_3d_point3(self, points_3d_ids, image):
        points_2d = []
        for point_3d in points_3d_ids:
            point_2d_idx = self.find_3D_point_idx_in_image(point_3d, image)
            point_2d = image.xys[point_2d_idx]
            points_2d.append(point_2d)
        return np.array(points_2d)

    def find_3D_point_idx_in_image(self, point_3d_id, image):
        for i, point_3d_id_in_image in enumerate(image.point3D_ids):
            if point_3d_id_in_image == point_3d_id:
                return i
        return -1

    # def reduce_labels_per_sequence(self, label_path, max_images=1):
    def reduce_labels_per_sequence(self, annotations_per_src_image, max_images=1, sampling_scheme="1_per_cam"):
        # label_files = os.listdir(label_path)
        # # print("Num label files:", len(label_files))

        # for label_file in label_files:
        #     with open(label_path + label_file) as f:
        #         labels = json.load(f)

            # print("len imgs", label_path + label_file, len(labels["images"]), labels["annotations"])
        labels = self.labelWriter.annotation_dict
            
        # TODO: Maybe remove following line if num src annots == num target annots works
        # if len(labels["images"]) <= max_images:
        #     return

        # sort the images by the number of shared features
        images = labels["images"]
        annotations = labels["annotations"]
        
        # print("len imgs", len(images), len(annotations))
        image_id_to_num_shared_features = {}
        for annotation in annotations:
            if annotation["image_id"] not in image_id_to_num_shared_features:
                image_id_to_num_shared_features[annotation["image_id"]] = 0
            image_id_to_num_shared_features[annotation["image_id"]] += len(annotation["shared_features"])
        
        sorted_images = sorted(images, key=lambda x: image_id_to_num_shared_features[x["id"]], reverse=True)



        # for each image, get the number of annotations in it
        annotations_per_image = {}
        for annotation in annotations:
            if annotation["image_id"] not in annotations_per_image:
                annotations_per_image[annotation["image_id"]] = []
            annotations_per_image[annotation["image_id"]].append(annotation)

        if sampling_scheme != "no_verification":
            # check for each image if it contains less annotations than the max of its source images. if so, remove it from the list
            imgs_to_remove = []
            src_images_per_image = {}
            for image in sorted_images:
                # print("image", image)
                src_images = [annotation["src_ids"] for annotation in annotations_per_image[image["id"]]]
                # flatten src_images
                src_images = [src_image for src_image_list in src_images for src_image in src_image_list]
                flattened_src_images = []
                for src_image_list in src_images:
                    flattened_src_images += src_image_list
                src_images = flattened_src_images
                src_images_per_image[image["id"]] = src_images

                # src_images = [self.db_img_id_to_label_img_id[src_id] for anno in annotations["src_ids"]]
                # src_images = [self.db_img_id_to_label_img_id[src_id] for src_id in image["src_ids"]]
                # print("self.db_img_id_to_label_img_id", self.db_img_id_to_label_img_id)
                # print("src_images", src_images)
                # print("annotations_per_src_image", annotations_per_src_image)
                
                max_annotations = max([len(annotations_per_src_image[src_image]) for src_image in src_images])
                print(len(annotations_per_image[image["id"]]), " vs ", max_annotations)
                # print the bounding boxes of each annotation
                if len(annotations_per_image[image["id"]]) < max_annotations:
                    imgs_to_remove.append(image["id"])
            # print("imgs_to_remove", imgs_to_remove)
            sorted_images = [image for image in sorted_images if image["id"] not in imgs_to_remove]

        # images = sampled_images

        # print("len imgs", len(images), len(annotations))
        # print("num_images_per_individual_settings", num_images_per_individual_settings)


        sorted_immages_cam_3 = [img for img in sorted_images if "camera3" in img["file_name"]]
        sorted_images_cam_5 = [img for img in sorted_images if "camera5" in img["file_name"]]
        sorted_images_no_cam = [img for img in sorted_images if "camera3" not in img["file_name"] and "camera5" not in img["file_name"]]
        images_cam_3 = sorted_immages_cam_3[:max_images]
        images_cam_5 = sorted_images_cam_5[:max_images]
        images_no_cam = sorted_images_no_cam[:max_images]

        if sampling_scheme == "1_per_cam":
            images = images_cam_3 + images_cam_5 + images_no_cam
        elif sampling_scheme == "cam_5_only":
            images = images_cam_5 + images_no_cam # + images_cam_3
        elif sampling_scheme == "improved":
            # we want to collect at most max_images images from the sorted_images for each camera and src img combination and time
            num_images_per_individual_settings = {}
            sampled_images = []
            min_time_diff = 10 # at least 5 seconds should be between 2 similar images
            for image in sorted_images:
                img_time = image["file_name"].split('/')[-1][:-4].replace("_", ".")
                # print("img_time", img_time)
                img_time = float(img_time)
                is_camera_3 = "camera3" in image
                src_ids = tuple(src_images_per_image[image["id"]])
                img_triple = (img_time, is_camera_3, src_ids)
                # print("img_triple", img_triple, type(img_triple))
                has_been_sampled = False
                for collected_img in num_images_per_individual_settings:
                    if abs(collected_img[0] - img_time) < min_time_diff and collected_img[1] == is_camera_3 and collected_img[2] == src_ids:
                        if num_images_per_individual_settings[collected_img] < max_images:
                            num_images_per_individual_settings[collected_img] += 1
                            sampled_images.append(image)
                            has_been_sampled = True
                            break
                        else:
                            has_been_sampled = True
                            break
                if not has_been_sampled:
                    # print("img_tripleee", img_triple, type(img_triple))
                    num_images_per_individual_settings[img_triple] = 1
                    sampled_images.append(image)

            images = sampled_images
        elif sampling_scheme == "no_verification":
            images = images_cam_3 + images_cam_5 + images_no_cam
            # images = sorted_images[:max_images]
        else:
            # throw an exception
            raise Exception("Sampling scheme not implemented: " + sampling_scheme)

        # remove annotations that are not in the top max_images
        annotations = [annotation for annotation in annotations if annotation["image_id"] in [image["id"] for image in images]]

        labels["images"] = images
        labels["annotations"] = annotations


        print("len_imgs and len_annotations", len(images), len(annotations))
        self.labelWriter.annotation_dict = labels
        # TODO: save the reduced labels to be merged
        # with open(self.label_path + self.label_file, "w") as f:
        #     json.dump(labels, f)
        # print("Reduced labels for", label_path + label_file)

    def save_annotation_images(self, labels, output_path, draw_label=True, keep_label_img_name=True, overwrite=False):
        images = labels["images"]
        annotations = labels["annotations"]

        if len(images) == 0 or len(annotations) == 0:
            if draw_label:
                BoxGenerator.no_img_generated_count += 1
            print("No images or annotations have been generated for ", output_path)
            return
        if draw_label:
            BoxGenerator.img_generated_count += 1
        print("Saving images for", output_path)
        # print(images[0].keys(), annotations[0].keys())

        if overwrite and os.path.exists(output_path):
            # remove the folder using shutil if it exists
            shutil.rmtree(output_path)
        
        # create a folder label_visualizations
        os.makedirs(output_path, exist_ok=True)

        all_boxes = {}
        for annotation in annotations:
            if annotation["image_id"] not in all_boxes:
                all_boxes[annotation["image_id"]] = []
            all_boxes[annotation["image_id"]].append(annotation["bbox"] + [annotation["src_ids"][0], annotation.get("shared_features", [])])

        img_id_to_name = {}
        for image in images:
            img_id_to_name[image["id"]] = image["file_name"]

        # # save all images for which exists labels in the folder label_visualizations with the labels drawn in
        for img_id, boxes in all_boxes.items():
            img_name = img_id_to_name[img_id]
            # print("img_name", img_name)
            if not keep_label_img_name:
                img_name = img_name.split('/')[-1]

            output_name = f"{output_path}/{sequence_id}_{img_id}_{len(boxes)}"
            # check if the image already exists
            if os.path.exists(output_name + ".png") and not overwrite:
                continue

            img_path = os.path.join(self.images_path, img_name)

            if draw_label:
                img = plt.imread(img_path)
                fig, axis = plt.subplots(1, figsize=(10,10))
                # dpi = 200
                # fig, axis = plt.subplots(figsize=(1280/80, 720/80))  # Divide by 80 to get the inches size
                axis.imshow(img)

                # only show the image, no axis
                plt.axis('off')

                all_src_ids = set([])
                # draw the bounding boxes
                for box in boxes:
                    x, y, w, h, src_ids, shared_features = box
                    # print("SSSS", src_ids)
                    if type(src_ids) == int:
                        src_ids = [src_ids]
                    all_src_ids.update(src_ids)
                    # print("all_src_ids", all_src_ids)
                    # TODO: uncomment next line
                    axis.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', lw=1))
                    # show the shared features as points in the box
                    img_keypoints = np.array(shared_features)
                    # draw the keypoints
                    # axis.scatter(img_keypoints[:,0], img_keypoints[:,1], s=1, c='orange', marker='o')

                    # put a label next to the box, with the num of shared_features for this box
                    # TODO: uncomment next line
                    # axis.text(x, y, f"{len(shared_features)}")
                
                # axis.set_title(f"img_id: {img_id}, src_ids: {all_src_ids}")

                output_name += f"_src_{all_src_ids}".replace(" ", "")

                # draw the keypoints
                # axis.scatter(img_keypoints[:,0], img_keypoints[:,1], s=5, c='orange', marker='o')

                axis.set_xlim(0, img.shape[1])
                axis.set_ylim(img.shape[0], 0)

                # save the image
                print("ouput name", output_name)
                plt.savefig(output_name, bbox_inches='tight', pad_inches=0, dpi=250) # , dpi=500
                # also save the image as a pdf with reduced size
                pdf_name = output_name + ".pdf"
                plt.savefig(pdf_name, bbox_inches='tight', pad_inches=0, dpi=150)
                plt.close()
            else:
                full_output_path = f"{output_path}/{img_name}"
                # if the output path does not exist, create it
                if not os.path.exists(os.path.dirname(full_output_path)):
                    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
                # copy the image to the output folder
                print("copying to ", full_output_path)
                shutil.copy(img_path, full_output_path)


if __name__ == "__main__":
    # use python box_generator.py (data_path) (sequence_ids) (re_identification_name) (-v)

    parser = argparse.ArgumentParser(description="Extract images and GPS from a rosbag.")
    parser.add_argument(
        "-p", "--data_path", default='../data', help="The path to the data folder")
    parser.add_argument(
        "-s", "--seq-ids", nargs='+', type=int, default=[-1,], help="Selected sequence IDs for which to generate labels, or -1 for all static sequences")
    all_folders = ["new/2021_02_18_06_28", "2021_03_25_14_04", "new/2021_04_05_14_35", "2021_05_18_14_02", "new/2021_06_05_12_08", "new/2021_07_07_06_41", "new/2021_08_09_06_30", "new/2021_09_15_06_28", "new/2021_10_15_18_16", "new/2021_11_02_12_59", "new/2021_12_15_12_54", "2022_01_21_14_04"]
    parser.add_argument(
        "-d", "--data_name", nargs='+', type=str, default=all_folders, help="The name of the folders for which labels should be generated")
    parser.add_argument(
        "-o", "--output_path", default='../outputs', help="The path where to save the outputs")
    parser.add_argument(
        "-v", "--visualize", action='store_true', help="Whether to visualize the generated labels")
    parser.add_argument(
        "-x", "--dont_save_labels", action='store_true', help="Whether to discard the generated labels")
    parser.add_argument(
        "-n", "--num_labels", default=1, help="The number of images to generate per sequence per camera")
    parser.add_argument(
        "-f", "--min_shared_features", default=3, help="The minimum number of shared features for a box to be generated")
    parser.add_argument(
        "-m", "--use_mean", action='store_true', help="Whether to use the mean of the shared features as the center of the box instead of the median")
    parser.add_argument(
        "-sampl", "--sampling_scheme", default='1_per_cam', help="Which images to sample")
    args = parser.parse_args()

    if len(args.data_path) > 0 and args.data_path[-1] != "/":
        args.data_path += "/"
    if len(args.output_path) > 0 and args.output_path[-1] != "/":
        args.output_path += "/"

    for data_name in args.data_name:
            

        # data_path = "../data/"
        # if len(sys.argv) >= 2:
        #     data_path = sys.argv[1]

        # sequence_ids = [-1]
        # if len(sys.argv) >= 3:
        #     sequence_ids = [int(sys.argv[2])]
        
        reconstruction_name = "label_reconstructions35"
        if args.seq_ids[0] == -1:
            sequence_ids = [0, 2, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 30, 33, 41, 43, 46, 
                            49, 53, 54, 55, 56, 57, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 
                            77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 94, 95, 96, 97, 98, 102, 
                            103, 104, 106, 107, 110, 111, 112, 113, 115, 116, 117, 118, 120, 121, 122, 125, 126, 
                            128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 145, 147, 148, 
                            149, 150, 153, 156, 160, 163, 164, 166, 168, 169, 170, 180, 181, 183]
        elif args.seq_ids[0] == -3:
            # print(os.listdir(os.path.join(data_path, reconstruction_name, "sequences")))
            # available_sequences = len(os.listdir(f"{data_path}{reconstruction_name}/sequences"))
            available_sequences = len(os.listdir(os.path.join(args.data_path, reconstruction_name, "sequences")))
            sequence_ids = list(range(available_sequences))
        elif args.seq_ids[0] == -2:
            # sequences with only one label per image:
            sequences_with_one_label_per_img = [0, 2, 7, 9, 12, 15, 17, 18, 19, 20, 23, 26, 43, 44, 50, 59, 60, 63, 
                67, 72, 80, 83, 87, 89, 91, 98, 113, 116, 118, 121, 126, 129, 130, 132, 135, 145, 149, 150, 151, 
                152, 156, 158, 159, 162, 166, 170, 172, 173, 178, 181, 182, 183]
            sequence_ids = sequences_with_one_label_per_img
        else:
            sequence_ids = args.seq_ids
        

        output_path_vis = args.output_path + "vis/" + data_name + "/"

        # draw_visualizations = False
        
        # if args.visualize:
        #     draw_visualizations = True

        draw_visualizations = args.visualize
        # if len(sys.argv) >= 5:
            # draw_visualizations = sys.argv[4] == "-v"

        output_path = "../outputs/"

        for sequence_id in sequence_ids:
            # label_file_name = "/sunny_2021_03_23_14_33_cam5_filtered_detection_ground_truth_labels(training dataset).json"
            label_file_name = "/sunny_reduced_static.json"

            # os.path.join(images_path, re_identification_name)
            print(f"Processing sequence {sequence_id}")

            boxGenerator = BoxGenerator(args.data_path, "label_reconstructions35", label_file_name, sequence_id, data_name, int(args.min_shared_features), True, debug=False)
            boxGenerator.generate_boxes(int(args.num_labels), args.use_mean, args.sampling_scheme)
            # continue

            path_to_orig_data = "../data/Waste_Bin_Detection_Dataset/sunny_2021_03_23_14_33_cam5_filtered (training dataset images and ground truths)/images/"

            # print("out ", f"{output_path}label_visualizations_{boxGenerator.sequence_id}")

            # boxGenerator.save_annotation_images(boxGenerator.labelWriter.annotation_dict, f"{output_path}/{re_identification_name}", draw_label=False, keep_label_img_name=True, overwrite=False)

            # # save_annotations_as_images(labels, output_path + f"/label_visualizations_orig_names", path_to_orig_data, False)
            # if draw_visualizations:
            #     boxGenerator.save_annotation_images(boxGenerator.labelWriter.annotation_dict, f"{output_path_vis}label_visualizations_single_{boxGenerator.sequence_id}", draw_label=True, keep_label_img_name=True, overwrite=True)

            # boxGenerator.reduce_labels_per_sequence(boxGenerator.generated_label_path, 3)
            # boxGenerator.reduce_labels_per_sequence(int(args.num_labels))
            print(f"Reduced labels to {int(args.num_labels)}")
            # TODO: uncomment next line
            if not args.dont_save_labels:
                boxGenerator.save_annotation_images(boxGenerator.labelWriter.annotation_dict, f"{output_path}/{data_name}", draw_label=False, keep_label_img_name=True, overwrite=False)

            if draw_visualizations:
                save_name = f"{output_path_vis}label_visualizations_reduced_{boxGenerator.sequence_id}"
                print("Saving visualizations at ", save_name)
                boxGenerator.save_annotation_images(boxGenerator.labelWriter.annotation_dict, save_name, draw_label=True, keep_label_img_name=True, overwrite=True)

        print("No sparse model found: ", BoxGenerator.no_sparse_model_count)
        print("No feature points: ", BoxGenerator.no_features_count)
        print("features found: ", BoxGenerator.features_count)
        print("Boxes generated: ", BoxGenerator.box_generated_count)
        print("No boxes generated: ", BoxGenerator.no_box_generated_count)
        print("Images generated: ", BoxGenerator.img_generated_count)
        print("No images generated: ", BoxGenerator.no_img_generated_count)
        print("Labels generated: ", BoxGenerator.generated_labels_per_sequence)
        # exit(1) # TODO: remove line

        print("Finished generating boxes", boxGenerator.generated_label_path, f"{output_path}{data_name}")

        if boxGenerator is not None and not args.dont_save_labels:
            
            from merge_COCO import merge_coco
            merge_coco(boxGenerator.generated_label_path, f"{output_path}{data_name}") # , boxGenerator.reconstruction_folder)

# TODO:
# Count label quality pre and post decimation to same num annotations
    # DONE generate images wo labels for better visibility
    # Decide which labels to count
# DONE Get sequences that contain static bins
# Check which settings yield the best results
# Visualize keypoints/shared features

# With reduced label set:
# No sparse model found:  81
# No feature points:  49
# features found:  57

# With full label set:
# No sparse model found:  81
# No feature points:  7
# features found:  99

# python box_generator.py ../data/ -1 new/2021_02_18_06_28 -v



    # def calculate_new_box(common_keypoints_source, common_keypoints, original_box):
    #     median_base = np.median(common_keypoints_source, axis=0)
    #     print("Median base", median_base)
    #     median_new = np.median(common_keypoints, axis=0)
    #     box_to_base_median = median_base[:2] - original_box[:2]
    #     return [median_new[0] - box_to_base_median[0],
    #                 median_new[1] - box_to_base_median[1],
    #                 original_box[2], original_box[3]]

    # def AffinityKeypoint2ScaleX(self, keypoint):
    #     x, y, a11, a12, a21, a22 = keypoint
    #     return np.sqrt(a11 * a11 + a21 * a21)

    # def AffinityKeypoint2ScaleY(self, keypoint):
    #     x, y, a11, a12, a21, a22 = keypoint
    #     return np.sqrt(a12 * a12 + a22 * a22)

    # def calculate_new_box(self, common_keypoints_source, common_keypoints, original_box):
    #     median_base = np.median(common_keypoints_source, axis=0)
    #     print("Median base", median_base)
    #     median_new = np.median(common_keypoints, axis=0)
    #     box_to_base_median = median_base[:2] - original_box[:2]
    #     new_box_widths = []
    #     new_box_heights = []
    #     new_box_x_scalings = []
    #     new_box_y_scalings = []
    #     for common_keypoint_src, common_keypoint in zip(common_keypoints_source, common_keypoints):
    #         new_box_x_scalings.append(self.AffinityKeypoint2ScaleX(common_keypoint) / self.AffinityKeypoint2ScaleX(common_keypoint_src))
    #         new_box_y_scalings.append(self.AffinityKeypoint2ScaleY(common_keypoint) / self.AffinityKeypoint2ScaleY(common_keypoint_src))
    #         # new_box_widths.append(new_box_x_scalings[-1] * original_box[2])
    #         # new_box_heights.append(new_box_y_scalings[-1] * original_box[3])
    #         # new_box_widths.append(AffinityKeypoint2ScaleX(common_keypoint) / AffinityKeypoint2ScaleX(common_keypoint_src) * original_box[2])
    #         # new_box_heights.append(AffinityKeypoint2ScaleY(common_keypoint) / AffinityKeypoint2ScaleY(common_keypoint_src) * original_box[3])
    #     new_box_scale_x = np.median(new_box_x_scalings)
    #     new_box_scale_y = np.median(new_box_y_scalings)
        
    #     new_box_width = new_box_scale_x * original_box[2]
    #     new_box_height = new_box_scale_y * original_box[3]
    #     # new_box_width = AffinityKeypoint2ScaleX(common_keypoints[0]) / AffinityKeypoint2ScaleX(common_keypoints_source[0]) * original_box[2]
    #     # new_box_height = AffinityKeypoint2ScaleY(common_keypoints[0]) / AffinityKeypoint2ScaleY(common_keypoints_source[0]) * original_box[3]
    #     return [median_new[0] - box_to_base_median[0],
    #                 median_new[1] - box_to_base_median[1],
    #                 new_box_width, new_box_height]

    # def calculate_new_box(common_keypoints_source, common_keypoints, original_box):
    #     box_corners = np.array([[original_box[0], original_box[1]],
    #                             [original_box[0] + original_box[2], original_box[1]],
    #                             [original_box[0], original_box[1] + original_box[3]],
    #                             [original_box[0] + original_box[2], original_box[1] + original_box[3]]])
    #     transformed_boxes = []
    #     for common_src_keypoint in common_keypoints_source:
    #         # common_src_keypoint is of form [x, y, a11, a12, a21, a22]
    #         affine_transformation_src = np.array([[common_src_keypoint[2], common_src_keypoint[3]],
    #                                             [common_src_keypoint[4], common_src_keypoint[5]]])
    #         affine_transformation = np.array([[common_keypoints[0][2], common_keypoints[0][3]],
    #                                         [common_keypoints[0][4], common_keypoints[0][5]]])
                                            
    #         print("Shape box corners", box_corners.shape)
    #         print("Shape affine transformation", affine_transformation.shape)
    #         print("Src affine vs inv affine", affine_transformation_src, np.linalg.inv(affine_transformation), " vs affine ", affine_transformation)
    #         print("Src point transformed vs target point", np.matmul(common_src_keypoint[:2], affine_transformation_src), common_keypoints[0])
    #         # affine_transformation_src_inv = np.linalg.inv(affine_transformation_src)
    #         # transformed_box = box_corners
    #         # transpose the affine transformation
    #         # affine_transformation = np.transpose(affine_transformation)
    #         # # invert the affine transformation
    #         # affine_transformation = np.linalg.inv(affine_transformation)
    #         # transformed_box = np.matmul(box_corners, 
    #         #                             affine_transformation_src_inv) # + np.array([common_src_keypoint[0], common_src_keypoint[1]])
    #         # transformed_box = np.matmul(transformed_box, 
    #         #                             affine_transformation) # + np.array([common_keypoints[0][0], common_keypoints[0][1]])
    #         print("Transformed box", transformed_box.shape)
    #         transformed_boxes.append(transformed_box)
    #     transformed_boxes = np.array(transformed_boxes)
    #     transformed_box = transformed_boxes[0]
    #     return [transformed_box[0][0], transformed_box[0][1],
    #      transformed_box[3][0] - transformed_box[0][0], transformed_box[3][1] - transformed_box[0][1]]


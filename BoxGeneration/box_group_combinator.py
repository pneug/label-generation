def combine_grouped_boxes(labelWriter):
    annotation_box_dict = get_annotation_box_dict(labelWriter)
    reduced_box_dict = get_reduced_box_dict(annotation_box_dict)
    labelWriter.update_annotations(reduced_box_dict)

def get_annotation_box_dict(labelWriter):
    box_dict = {}
    for annotation in labelWriter.annotation_dict['annotations']:
        box_dict[annotation['image_id']] = box_dict.get(annotation['image_id'], []) + [(annotation["src_ids"], annotation["shared_features"], annotation['bbox'])]
        # print("src_idd", annotation["src_ids"])
    return box_dict

def get_reduced_box_dict(new_box_dict):
    reduced_box_dict = {}
    # dict with img_id as key and dict with box infos as value
    # box infos: new:box_id: running_median_box, num_points, all_boxes
    for img_id, boxes in new_box_dict.items():
        # max_shared_features = 0
        # print("boxes", boxes)
        for box in boxes:
            src_id, shared_features, box = box
            if img_id in reduced_box_dict:
                added_box = False
                for box_groups in reduced_box_dict[img_id]:
                    # check if the box overlaps with any of the existing boxes
                    iou = intersection_over_union(box_groups['running_median_box'], box)
                    
                    if iou > 0.01:
                        # merge the boxes
                        new_median_box = calculate_new_median_box(box_groups['running_median_box'], box, len(box_groups['all_boxes']))
                        box_groups['running_median_box'] = new_median_box
                        box_groups['all_boxes'].append(box)
                        box_groups['src_ids'].append(src_id)
                        if len(shared_features) > len(box_groups.get('shared_features', [])):
                            box_groups['best_box'] = box
                            box_groups['shared_features'] = shared_features

                        added_box = True
                        break

                if not added_box:
                    new_box_group = {'running_median_box': box, 'all_boxes': [box], 'src_ids': [src_id], 
                                     'best_box': box, 'shared_features': shared_features}
                    reduced_box_dict[img_id] = reduced_box_dict.get(img_id, []) + [new_box_group]
            else:
                reduced_box_dict[img_id] = [{'running_median_box': box, 'all_boxes': [box], 'src_ids': [src_id], 
                                            'best_box': box, 'shared_features': shared_features}]   
    return reduced_box_dict     

# def get_max_feature_box()

def calculate_new_median_box(curr_median_box, new_box, curr_box_weight=1):
    # calculate the new median box
    new_median_box = []
    for i in range(len(curr_median_box)):
        new_median_box.append((curr_median_box[i] * curr_box_weight + new_box[i]) / (curr_box_weight + 1))
    return new_median_box

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
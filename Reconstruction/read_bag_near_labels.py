# -*- coding: utf-8 -*-

"""
Extract images and GPS from a rosbag.
"""

import os
from os.path import isfile, join
import argparse

import cv2

import rosbag, rospy
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
from exif import set_gps_location

from time import perf_counter
from read_annotations import get_annotation_groups
import yaml

from tqdm import tqdm

def write_images(img_buffer, args):
    for img in img_buffer:
        image_dir, cv_img, LAST_GPS = img
        # img_buffer.append(image_dir, cv_img, LAST_GPS)
        cv2.imwrite(image_dir, cv_img)
        if args.gps_save:
            set_gps_location(image_dir, LAST_GPS.latitude, LAST_GPS.longitude, LAST_GPS.altitude)

def main():
    parser = argparse.ArgumentParser(description="Extract images and GPS from a rosbag.")
    parser.add_argument(
        "-f", "--folder", default='.', help="The folder from which all Ros Bags should get read")
    parser.add_argument(
        "-i", "--input", nargs='+', type=str, default=[], help="Input ROS bags")
    #parser.add_argument(
    #    "-i", "--input", default='./test.bag', help="Input ROS bag")
    parser.add_argument(
        "-c", "--cam-id", nargs='+', type=int, default=[3,], help="Selected camera IDs to extract")
    parser.add_argument(
        "-o", "--output", default='./output', help="Output dir")
    parser.add_argument(
        "-g", "--gps-save", action='store_true', help="Whether to save GPS as exif info of the images")
    parser.add_argument(
        "-n", "--num_images", type=int, default=0, help="Amount of frames that should be extracted")
    parser.add_argument(
        "-s", "--split", action='store_true', help="Whether to split images from different cameras into subfolders")
    parser.add_argument(
        "-v", "--velocity", type=float, default=0., help="The min velocity for images to be extracted")
    # parser.add_argument(
        # "-r", "--recurse", action='store_true', help="Extra")
    parser.add_argument(
        "-l", "--labels", default='', help="If given, only images near images with labels will be extracted")
    parser.add_argument(
        "-si", "--startBagIndex", type=int, default=0, help="The id of the first bag to be processed")
    parser.add_argument(
        "-ei", "--endBagIndex", type=int, default=0, help="The id of the last bag to be processed")
    parser.add_argument(
        "-d", "--distance", type=float, default=-1., help="The distance between samples in meters")
    
    args = parser.parse_args()

    bag_files = args.input
    folder = args.folder
    output_dir = args.output
    num_images = args.num_images
    split = args.split
    min_velocity = args.velocity

    print("args.endBagIndex ", args.endBagIndex)
    if args.endBagIndex == 0:
        args.endBagIndex = -1

    if len(bag_files) == 0:
        bag_files = sorted([join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f)) and f[-4:] == ".bag"])

    if args.endBagIndex < 0:
        args.endBagIndex = len(bag_files)

    bag_files = bag_files[args.startBagIndex:args.endBagIndex]
    print("start id ", args.startBagIndex, " end id ", args.endBagIndex)

    groups, groupNames = get_annotation_groups(args.labels)
    # print the image ids for the first 10 groups
    for i in range(10):
        print(groups[i])

    samples = {}

    for group_num, group in enumerate(groups):
        # get all the images between the first and last image of the group
        start_time = group[0] - 2
        end_time = group[-1] + 2

        used_bags = []

        # find all the bags that contain images between the start time and the end time
        print("start_time ", start_time, " end_time ", end_time)
        # last_bag = bag_files[0]
        #if bag_name_to_time(last_bag) > start_time:
        #    print("skipping bag")
        #    continue

        for i in range(len(bag_files) - 1):
            bag = bag_files[i]
            # check if the bag is between the start and end time
            # The bag name is the time of the first image in the bag
            # Each bag goes until the next bag begins
            # bagContainsImages = bag_name_to_time(bag_files[i]) < start_time and bag_name_to_time(bag_files[i+1]) > start_time
            # print("bag ", bag)
            if bag_name_to_time(bag) <= end_time and bag_name_to_time(bag_files[i+1]) >= start_time:
                used_bags.append(bag)
            #if bag_name_to_time(bag) > end_time or bag_name_to_time(bag_files[i+1]) > end_time:
            #    break
            # last_bag = bag
        
        if bag_name_to_time(bag_files[-1]) <= end_time:
            used_bags.append(bag_files[-1])

        print("used_bags ", used_bags)


        # stop the time 
        t1_start = perf_counter()
        extract(used_bags, output_dir + "/" + str(group_num), args.gps_save, args.cam_id, split, min_velocity, start_time, end_time, args.distance)
        t1_stop = perf_counter()
        print(f"Elapsed time for reading bag {group_num} of {len(groups)} in seconds:", t1_stop-t1_start)

def bag_name_to_time(bag_name):
    # bag name is e.g. bus_2021-03-23-14-33-58_0.bag
    date = bag_name.split('/')[-1]
    date = date.split('_')[1]
    # convert the date to common data structure
    import datetime
    date = datetime.datetime.strptime(date, '%Y-%m-%d-%H-%M-%S')
    # convert the date to seconds
    # date = date.timestamp()
    return date.timestamp()


def save_image(msg, bridge, topic, output_folder_dir, gps_save, LAST_GPS, split, samples):
    cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

    cam_seperator = '/' if split else '_'
    cam_name = topic[1:8]
    if split:
        os.makedirs(os.path.join(output_folder_dir, "images", cam_name), exist_ok=True)
    time_stamps = cam_seperator + '{:0>10d}_{:0>9d}'.format(t.secs, t.nsecs)
    image_filename = "images/" + cam_name + time_stamps + '.jpg' 
    image_dir = os.path.join(output_folder_dir, image_filename)
    
    # img_buffer.append((image_dir, cv_img, LAST_GPS))
    cv2.imwrite(image_dir, cv_img)
    if gps_save and LAST_GPS is not None:
        set_gps_location(image_dir, LAST_GPS.latitude, LAST_GPS.longitude, LAST_GPS.altitude)


def save_sampled_images(buffered_images, bridge, output_folder_dir, gps_save, LAST_GPS, split, sample_dist, samples, msg, t, LAST_GPS_TIME):
    # go through all buffered images and save them if there is not a sample nearby yet
    # round_factor = 0.00001 * sample_dist # 1m
    for sample_topic, sample_msg, sample_t in buffered_images:
        if LAST_GPS is None:
            continue
            # save_image(sample_msg, bridge, sample_msg, output_folder_dir, gps_save, LAST_GPS, split, samples)
        else:
            # interpolate the gps position by using the time of the sample and the last gps position and the current gps position, using milliseconds for accuracy
            interpolation_factor = (sample_t.to_sec() - LAST_GPS_TIME.to_sec()) / (t.to_sec() - LAST_GPS_TIME.to_sec())
            interpolated_loc = (LAST_GPS.latitude + (msg.latitude - LAST_GPS.latitude) * interpolation_factor,
                                LAST_GPS.longitude + (msg.longitude - LAST_GPS.longitude) * interpolation_factor)
            rounded_loc = (round(interpolated_loc[0], 5), round(interpolated_loc[1], 5))
            if not rounded_loc in samples:
                save_image(sample_msg, bridge, sample_topic, output_folder_dir, gps_save, LAST_GPS, split, samples)
                samples[rounded_loc] = 1
    buffered_images = ()

def extract(bag_files, output_dir, gps_save, cam_id, split, min_velocity, start_time, end_time, sample_dist, samples):
    os.makedirs(output_dir, exist_ok=True)

    topics = ['/fix'] if gps_save else []
    topics.append('/velocity')
    for cam_id in cam_id:
        topics.append('/camera{}/image_raw/compressed'.format(cam_id))

    bridge = CvBridge()

    bus_stopped = False

    for num, bag_file in enumerate(bag_files):
        if False: # len(bag_files) > 1:
            bag_name = bag_file.split('/')[-1][:-4]
            output_folder_dir = os.path.join(output_dir, bag_name)
            os.makedirs(output_folder_dir, exist_ok=True)
        else:
            output_folder_dir = output_dir
        print(num, " / ", len(bag_files))
        print("Extract images from {} for topics {}".format(bag_file, topics))

        bag = rosbag.Bag(bag_file, "r")
        # info_dict = yaml.load(bag._get_yaml_info())
        # print(info_dict)

        if gps_save:
            LAST_GPS = None # NavSatFix()
            print(LAST_GPS)

        velocity = 0
        buffered_images = []
        connection_filter = lambda topic, datatype, md5sum, msg_def, header: topic in topics
        # read out the bag and use tqdm to show the progress
        for topic, msg, t in tqdm(bag.read_messages(topics=topics, start_time=rospy.Time(start_time), end_time=rospy.Time(end_time), connection_filter=connection_filter)):
        # for topic, msg, t in bag.read_messages(topics=topics, start_time=rospy.Time(0), end_time=rospy.Time(0.0001), connection_filter=connection_filter):
            if 'velocity' in topic:
                # print("Velocity: ", msg.velocity, min_velocity, '{:0>10d}_{:0>9d}'.format(t.secs, t.nsecs))
                velocity = msg.velocity
                bus_stopped = velocity <= min_velocity 
                    
            elif 'image_raw' in topic and not bus_stopped:
                if sample_dist > 0:
                    buffered_images.append((topic, msg, t))
                save_image(msg, bridge, topic, output_folder_dir, gps_save, LAST_GPS, split, samples)

            elif 'fix' in topic:
                if sample_dist > 0:
                    save_sampled_images(buffered_images, bridge, output_folder_dir, gps_save, LAST_GPS, split, sample_dist, samples, msg, t, LAST_GPS_TIME)
                else:
                    # if not is_sample_nearby(msg.latitude, msg.longitude, round_factor):
                    save_image(msg, bridge, topic, output_folder_dir, gps_save, LAST_GPS, split, samples)

                LAST_GPS = msg
                LAST_GPS_TIME = t
        # if sample_dist > 0:
        #     save_sampled_images(buffered_images, bridge, output_folder_dir, gps_save, LAST_GPS, split, sample_dist, samples, msg, t, LAST_GPS_TIME)
        bag.close()

    return

if __name__ == '__main__':
    main()
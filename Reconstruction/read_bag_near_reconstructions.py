# -*- coding: utf-8 -*-

"""
Extract images and GPS from a rosbag.
"""

import os
from os.path import isfile, join
import argparse

import cv2
import numpy as np

import rosbag, rospy
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
from exif import set_gps_location

from time import perf_counter
from read_annotations import get_annotation_groups
import yaml

from tqdm import tqdm
import message_filters

sampled_img_locations = []
unsampled_img_locations = []

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
        "-f", "--folder", nargs='+', type=str, default=[], help="The folders from which all Ros Bags should get read")
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
    output_dir = args.output
    num_images = args.num_images
    split = args.split
    min_velocity = args.velocity

    print("args.endBagIndex ", args.endBagIndex)
    if args.endBagIndex == 0:
        args.endBagIndex = -1

    for folder in args.folder:
        print("processing folder ", folder)

        if len(bag_files) == 0:
            bag_files = sorted([join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f)) and f[-4:] == ".bag"])

        if args.endBagIndex < 0:
            args.endBagIndex = len(bag_files)

        bag_files = bag_files[args.startBagIndex:args.endBagIndex]
        print("start id ", args.startBagIndex, " end id ", args.endBagIndex)

        # stop the time 
        t1_start = perf_counter()
        extract(bag_files, output_dir, args.gps_save, args.cam_id, split, min_velocity, args.distance)
        t1_stop = perf_counter()
        print(f"Elapsed time for reading bag in seconds:", t1_stop-t1_start)

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


def save_image(msg, bridge, topic, t, output_folder_dir, gps_save, LAST_GPS, split):
    locations = read_sequence_locations("")
    for seq_num, location in enumerate(locations):
        box_extension = 0.0 # 0.0002 # 20m
        minLat, maxLat, minLon, maxLon = location
        if LAST_GPS is None:#
            continue
        if LAST_GPS.latitude < minLat - box_extension or LAST_GPS.latitude > maxLat + box_extension or LAST_GPS.longitude < minLon - box_extension or LAST_GPS.longitude > maxLon + box_extension:
            continue

        cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cam_seperator = '/' if split else '_'
        cam_name = topic[1:8]
        if split:
            os.makedirs(os.path.join(output_folder_dir, str(seq_num), "images", cam_name), exist_ok=True)
        else:
            os.makedirs(os.path.join(output_folder_dir, str(seq_num), "images"), exist_ok=True)

        time_stamps = cam_seperator + '{:0>10d}_{:0>9d}'.format(t.secs, t.nsecs)
        image_filename = str(seq_num) + "/images/" + cam_name + time_stamps + '.jpg' 
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
            # save_image(sample_msg, bridge, sample_msg, sample_t, output_folder_dir, gps_save, LAST_GPS, split, samples)
        else:
            # interpolate the gps position by using the time of the sample and the last gps position and the current gps position, using milliseconds for accuracy
            interpolation_factor = (sample_t.to_sec() - LAST_GPS_TIME.to_sec()) / (t.to_sec() - LAST_GPS_TIME.to_sec())
            interpolated_loc = (LAST_GPS.latitude + (msg.latitude - LAST_GPS.latitude) * interpolation_factor,
                                LAST_GPS.longitude + (msg.longitude - LAST_GPS.longitude) * interpolation_factor)
            rounded_loc = (round(interpolated_loc[0], 5), round(interpolated_loc[1], 5)) # 1m is 5 digits
            # print(len(samples), "len samples")
            if not rounded_loc in samples:
                sampled_img_locations.append(interpolated_loc)
                save_image(sample_msg, bridge, sample_topic, sample_t, output_folder_dir, gps_save, LAST_GPS, split)
                samples[rounded_loc] = 1
                print("saved sample", len(samples), "at", rounded_loc)
            else:
                unsampled_img_locations.append(interpolated_loc)
                save_image(sample_msg, bridge, sample_topic, sample_t, output_folder_dir + "_unsampled", gps_save, LAST_GPS, split)
                print("Sample already exists at", rounded_loc, len(samples))
    buffered_images.clear()

def save_buffered_images(buffered_images, bridge, output_folder_dir, gps_save, LAST_GPS, split, bus_stopped, t):
    count = 0
    # go through all buffered images and save them
    for sample_topic, sample_msg, sample_t in buffered_images:
        # sampled_img_locations.append(interpolated_loc)
        if sample_t <= t - rospy.Duration(2):
            count += 1
            if not bus_stopped:
                save_image(sample_msg, bridge, sample_topic, sample_t, output_folder_dir, gps_save, LAST_GPS, split)
        # save_image(sample_msg, bridge, sample_topic, sample_t, output_folder_dir, gps_save, LAST_GPS, split)
    # buffered_images.clear()
    # print("Count", count, len(buffered_images))
    buffered_images = buffered_images[count:]
    # print("Cound", count, len(buffered_images))
    return buffered_images


def extract(bag_files, output_dir, gps_save, cam_id, split, min_velocity, sample_dist):
    os.makedirs(output_dir, exist_ok=True)

    topics = ['/fix'] if gps_save else []
    topics.append('/velocity')
    for cam_id in cam_id:
        topics.append('/camera{}/image_raw/compressed'.format(cam_id))

    bridge = CvBridge()

    count = 0
    bus_stopped = False
    buffered_images = []

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

        # if gps_save:
        LAST_GPS = None # NavSatFix()
        LAST_GPS_TIME = None
            # print(LAST_GPS)

        velocity = 0
        samples = {}
        # filter = message_filters.TimeSynchronizer([image_sub, info_sub], 10)
        connection_filter = lambda topic, datatype, md5sum, msg_def, header: topic in topics
        # read out the bag and use tqdm to show the progress
        for topic, msg, t in tqdm(bag.read_messages(topics=topics, connection_filter=connection_filter)):
        # for topic, msg, t in bag.read_messages(topics=topics, start_time=rospy.Time(0), end_time=rospy.Time(0.0001), connection_filter=connection_filter):
            
            # print("topic ", topic, " t ", t)
            if 'velocity' in topic:
                # print("Velocity: ", msg.velocity, min_velocity, '{:0>10d}_{:0>9d}'.format(t.secs, t.nsecs))
                last_velocity = velocity
                velocity = msg.velocity
                bus_stopped = velocity < min_velocity 
                # print("Min velocuty: ", min_velocity, "Velocity: ", velocity, "Bus stopped: ", bus_stopped, "Buffered images: ", len(buffered_images))
                # if not bus_stopped:
                buffered_images = save_buffered_images(buffered_images, bridge, output_folder_dir, gps_save, LAST_GPS, split, bus_stopped, t)
                # else:
                #     buffered_images.clear()   
            elif 'image_raw' in topic: #  and not bus_stopped:
                #if LAST_GPS is None:
                #    continue

                count += 1
                # if count > 180:# 180:
                #     break

                # if count < 0:# 140:
                #     continue

                # print("sample_dist", sample_dist, "min_velocity", min_velocity, "velocity", velocity)
                if sample_dist > 0 or min_velocity > 0:
                    buffered_images.append((topic, msg, t))
                    # print(len(buffered_images))
                else:
                    save_image(msg, bridge, topic, t, output_folder_dir, gps_save, LAST_GPS, split)

            elif 'fix' in topic:
                if sample_dist > 0:
                    save_sampled_images(buffered_images, bridge, output_folder_dir, gps_save, LAST_GPS, split, sample_dist, samples, msg, t, LAST_GPS_TIME)
                #else:
                    # if not is_sample_nearby(msg.latitude, msg.longitude, round_factor):
                    #save_image(msg, bridge, topic, t, output_folder_dir, gps_save, LAST_GPS, split, samples)

                LAST_GPS = msg
                LAST_GPS_TIME = t

        # if min_velocity > 0 and not bus_stopped:
        buffered_images = save_buffered_images(buffered_images, bridge, output_folder_dir, gps_save, LAST_GPS, split, bus_stopped, t)
        # if sample_dist > 0:
            # save_sampled_images(buffered_images, bridge, output_folder_dir, gps_save, LAST_GPS, split, sample_dist, samples, msg, t, LAST_GPS_TIME)
        bag.close()

    print("sampled", sampled_img_locations)
    print("unsampled", unsampled_img_locations)
    return

def read_sequence_locations(data_path):
    with open("../data/group_locations.txt", "r") as f:
        locations = f.readlines()

    locations = [location.strip().split(", ") for location in locations]
    return np.array(locations, dtype=float)

if __name__ == '__main__':
    main()
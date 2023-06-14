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
        "-r", "--rate", type=int, default=1, help="Every r-th image will be extracted")
    
    args = parser.parse_args()

    bag_files = args.input
    folder = args.folder
    output_dir = args.output
    num_images = args.num_images
    split = args.split
    min_velocity = args.velocity

    extract(bag_files, output_dir, folder, num_images, args.gps_save, args.cam_id, split, min_velocity, args.rate)

def extract(bag_files, output_dir, folder, num_images, gps_save, cam_id, split, min_velocity, rate):
    os.makedirs(output_dir, exist_ok=True)

    topics = ['/fix'] if gps_save else []
    topics.append('/velocity')
    for cam_id in cam_id:
        topics.append('/camera{}/image_raw/compressed'.format(cam_id))

    if len(bag_files) == 0:
        bag_files = sorted([join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f)) and f[-4:] == ".bag"])

    bridge = CvBridge()

    bus_stopped = False
    rate_count = 0

    for num, bag_file in enumerate(bag_files):
        if len(bag_files) > 1:
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
            LAST_GPS = NavSatFix()
            print(LAST_GPS)

        velocity = 0
        for topic, msg, t in bag.read_messages(topics=topics):
            if 'velocity' in topic:
                # print("Velocity: ", msg.velocity, min_velocity, '{:0>10d}_{:0>9d}'.format(t.secs, t.nsecs))
                velocity = msg.velocity
                bus_stopped = velocity <= min_velocity 
                    
            elif 'image_raw' in topic and not bus_stopped:
                if rate_count % rate != 0:
                    rate_count += 1
                    continue
                rate_count += 1

                cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

                cam_seperator = '/' if split else '_'
                cam_name = topic[1:8]
                if split:
                    os.makedirs(os.path.join(output_folder_dir, cam_name), exist_ok=True)
                time_stamps = cam_seperator + '{:0>10d}_{:0>9d}'.format(t.secs, t.nsecs)
                image_filename = cam_name + time_stamps + '.jpg' 
                image_dir = os.path.join(output_folder_dir, image_filename)
                
                # img_buffer.append((image_dir, cv_img, LAST_GPS))
                cv2.imwrite(image_dir, cv_img)
                if gps_save:
                    set_gps_location(image_dir, LAST_GPS.latitude, LAST_GPS.longitude, LAST_GPS.altitude)

            elif 'fix' in topic:
                LAST_GPS = msg

        bag.close()

    return

if __name__ == '__main__':
    main()
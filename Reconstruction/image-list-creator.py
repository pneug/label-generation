# This script takes a path as an input and creates a file "image-list.txt" in this directory
# The file contains a list of all the images in the input path directory, one image per line

import os
import sys

if len(sys.argv) < 2:
    print("Please provide a path to a directory")
    exit(1)

path = sys.argv[1]
if not os.path.isdir(path):
    print("The provided path is not a directory")
    exit(1)

path_adjustment = ""
if len(sys.argv) > 2:
    path_adjustment = sys.argv[2]

def extract_dir(f, root_path, rel_path, path_adjustment):
    # find all directories and files in the current directory
    for file in os.listdir(os.path.join(root_path, rel_path)):
        if file.endswith(".jpg"):
                # append a new line with the path to the image
                f.write(os.path.join(path_adjustment, rel_path, file) + "\n")
                print("Added file ", os.path.join(rel_path, file), " to the list")
        elif os.path.isdir(os.path.join(root_path, rel_path, file)):
            print("dir ", file)
            extract_dir(f, root_path, os.path.join(rel_path, file), path_adjustment)
        

with open(os.path.join(path, "image-list.txt"), "w") as f:
    # clear the file
    f.write("")
    extract_dir(f, path, "", path_adjustment)

# with open(os.path.join(path, "image-list.txt"), "w") as f:
#     # open the dir and all subdirectories
#     extract_dir()
#     currPath = ""
#     while True:
#         for root, dirs, files in os.walk(path):
#             for file in files:
#                 if file.endswith(".jpg"):
#                     f.write(os.path.join(currPath, file))
#             for dir in dirs:
#                 currPath = os.path.join(currPath, dir)
#     for file in os.listdir(path):
#         if file.endswith(".jpg"):
#             f.write(file)
#             print("Added file " + file + " to the list")

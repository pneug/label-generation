import sys, os
import numpy as np

# import __file__
sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), "..", "colmap", "scripts", "python"))
from database import COLMAPDatabase, blob_to_array, image_ids_to_pair_id, pair_id_to_image_ids

class ColmapDatabase:
    def __init__(self, path):
        self.path = path
        self.db = COLMAPDatabase.connect(path)
        print("Sucessfully opened the database")

    def get_keypoints(self):
        return dict(
        (image_id, blob_to_array(data, np.float32, (-1, 6))) # 2
        for image_id, data in self.db.execute(
            "SELECT image_id, data FROM keypoints"))

    def get_matches(self):
        return dict(
            (pair_id_to_image_ids(pair_id),
            blob_to_array(data, np.uint32, (-1, 2)))
            for pair_id, data in self.db.execute(
                "SELECT pair_id, data FROM matches") if data is not None)

    def get_images(self):
        return dict(
            (image_id, {"name": name})
            for image_id, name in self.db.execute(
                "SELECT image_id, name FROM images"))

for d in $(find $DATA_PATH/label_reconstructions35/sequences -mindepth 1 -maxdepth 1 -type d | sort -n); do
    # commands to be executed for each directory
    echo "Processing directory: $d"
  
   DATASET_PATH=$d
   colmap feature_extractor \
      --database_path $DATASET_PATH/database.db \
      --image_path $DATASET_PATH/images \
      --ImageReader.camera_model OPENCV \
      --ImageReader.single_camera 1 \
      --SiftExtraction.estimate_affine_shape=true \
      --SiftExtraction.domain_size_pooling=true
   colmap exhaustive_matcher \
      --database_path $DATASET_PATH/database.db \
      --SiftMatching.guided_matching=true
   mkdir $DATASET_PATH/sparse
   colmap mapper \
       --database_path $DATASET_PATH/database.db \
       --image_path $DATASET_PATH/images \
       --export_path $DATASET_PATH/sparse \
       --Mapper.tri_ignore_two_view_tracks 0 \
       --Mapper.tri_min_angle 0.5

done

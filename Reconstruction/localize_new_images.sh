#!/bin/sh
set -e

source ~/.bashrc

for seq_path in $(find $1/unsupervised-data/$2 -mindepth 1 -maxdepth 1 -type d | sort -n); do
	echo "Processing directory: $seq_path"
	seq=$(basename "$seq_path")
	echo "Seq: $seq"
	export PROJECT_PATH="$1/label_reconstructions35/sequences/$seq"
	export IMAGE_PATH="$seq_path"
	export NEW_IMAGE_PATH="$PROJECT_PATH/images/$2"
	export VOCAB_PATH="$1"

	if test -f "$IMAGE_PATH/done.txt"; then
	# if test -f "$IMAGE_PATH/images/image-list.txt"; then
    		echo "Reconstruction for $seq exists, skipping."
		continue
	fi

	if [ "$(ls -1 $IMAGE_PATH/images/camera3/ | wc -l)" -gt $3 ]; then
		echo "$sec: more than 200 images, skipping for better performance"
		continue
	fi

	python image-list-creator.py $IMAGE_PATH/images $2/images

	mkdir $NEW_IMAGE_PATH -p
	cp -r $IMAGE_PATH/* $NEW_IMAGE_PATH

	# sleep 1

	colmap feature_extractor \
	    --database_path $PROJECT_PATH/database.db \
	    --image_path $PROJECT_PATH/images \
	    --image_list_path $NEW_IMAGE_PATH/images/image-list.txt \
	    --ImageReader.camera_model OPENCV \
	    --ImageReader.single_camera 1 \
	    --SiftExtraction.estimate_affine_shape=true \
	    --SiftExtraction.domain_size_pooling=true

	echo $PROJECT_PATH/database.db

	colmap exhaustive_matcher \
	    --database_path $PROJECT_PATH/database.db \
	    --SiftMatching.guided_matching=true \
	    # --SiftMatching.use_gpu 0

	#colmap vocab_tree_matcher \
	#    --database_path $PROJECT_PATH/database.db \
	#    --VocabTreeMatching.vocab_tree_path $VOCAB_PATH/vocab_tree_flickr100K_words32K.bin \
	#    --VocabTreeMatching.match_list_path $IMAGE_PATH/image-list.txt

	[ -d $NEW_IMAGE_PATH/new-sparse-model ] || mkdir $NEW_IMAGE_PATH/new-sparse-model -p

	echo $PROJECT_PATH/sparse/0/

	colmap image_registrator \
	    --database_path $PROJECT_PATH/database.db \
	    --input_path $PROJECT_PATH/sparse/0/ \
	    --output_path $NEW_IMAGE_PATH/new-sparse-model
	    # --import_path $PROJECT_PATH/sparse/0/ \
	    # --export_path $NEW_IMAGE_PATH/new-sparse-model

	colmap bundle_adjuster \
	    --input_path $NEW_IMAGE_PATH/new-sparse-model \
	    --output_path $NEW_IMAGE_PATH/new-sparse-model

	touch $IMAGE_PATH/done.txt

done

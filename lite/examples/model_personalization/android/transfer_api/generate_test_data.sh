#!/bin/bash

# This script generates the binary test_data.zip file used
# in transfer_api tests.

# How many samples should we keep for each class. There are 5 classes, so five times more
# samples are included in the archive.
NUM_FILES=75

rm -rf flower_photos test_data.zip

wget 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
tar xzf flower_photos.tgz
rm flower_photos.tgz

# Keep $NUM_FILES random samples for each class.
for dir in daisy dandelion roses sunflowers tulips; do
  find ./flower_photos/$dir -type f -print0 | sort -zR | tail -zn +$((NUM_FILES + 1)) | xargs -0 rm
done

cd flower_photos

# Randomly split every class into train/test with 80%/20% proportion.
echo -n >train.txt
echo -n >val.txt
for dir in daisy dandelion roses sunflowers tulips; do
  NUM_FILES=$(ls -1 $dir | wc -l)
  NUM_TRAIN=$((NUM_FILES / 5 * 4))
  ls -1 $dir | xargs -I{} mogrify -resize 224x224! $dir/{}
  ls -1 $dir | head -n $NUM_TRAIN | xargs -n 1 -I{} echo $dir/{},$dir >>train.txt
  ls -1 $dir | tail -n +$((NUM_TRAIN + 1)) | xargs -n 1 -I{} echo $dir/{},$dir >>val.txt
done

# Generate the archive and copy it to test assets.
zip -rq test_data *
cd ..

mkdir -p ./src/androidTest/assets/
mv flower_photos/test_data.zip ./src/androidTest/assets/

# Clean up.
rm -rf flower_photos

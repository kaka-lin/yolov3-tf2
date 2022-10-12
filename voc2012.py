import os
import glob
from datetime import datetime

import tqdm
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from tools.voc_to_tfrecord import process_image, parse_annot, create_tf_example

flags.DEFINE_string('data_dir', './data/VOCdevkit/VOC2012/', 'PASCAL VOC dataset')
flags.DEFINE_enum('split', 'train', ['train', 'val'], 'train or val dataset')
flags.DEFINE_string('output_dir', './data/', 'path of output dataset')
flags.DEFINE_string('classes', './model_data/voc2012_classes.txt', 'classes file')


def main(argv):
    classes_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", classes_map)

    output_file = os.path.join(
        FLAGS.output_dir, 'voc2012_{}.tfrecord'.format(FLAGS.split))

    with tf.io.TFRecordWriter(output_file) as writer:
        image_list = open(os.path.join(
            FLAGS.data_dir, 'ImageSets', 'Main', '%s.txt' % FLAGS.split)).read().splitlines()
        logging.info("Image list loaded: %d", len(image_list))

        counter = 0
        skipped = 0
        for image in tqdm.tqdm(image_list):
            image_file = os.path.join(FLAGS.data_dir, 'JPEGImages', '%s.jpg' % image)
            annot_file = os.path.join(FLAGS.data_dir, 'Annotations', '%s.xml' % image)

            # processes the image and parse the annotation
            error, image_string = process_image(image_file)
            image_info_list = parse_annot(annot_file, classes_map)

            if not error:
                # convert voc to `tf.Example`
                example = create_tf_example(image_string, image_info_list)

                # write the `tf.example` message to the TFRecord files
                writer.write(example.SerializeToString())
                counter += 1
            else:
                skipped += 1

    print('{} : Wrote {} images to {}'.format(
            datetime.now(), counter, output_file))


if __name__ == '__main__':
    app.run(main)

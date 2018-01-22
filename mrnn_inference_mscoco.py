import time
import sys
import os
import numpy as np
import logging
import json
import tensorflow as tf
from helper import *
from config import *
from tf_mrnn_decoder import mRNNDecoder


logger = logging.getLogger('ExpMscoco')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


flags = tf.flags

# CPU threads
flags.DEFINE_integer(
  "ses_threads", 4, 
  "Tensorflow CPU session threads to use"
  )
# GPU memoery usage
flags.DEFINE_float(
  "gpu_memory_fraction", 0.4, 
  "Fraction of GPU memory to use"
  )
# Model
flags.DEFINE_string(
  "model_root", 
  "./trained_models/ms_coco/", 
  "root of the tf mRNN model"
  )
flags.DEFINE_string(
  "model_name", 
  "mrnn_lstm.ckpt", 
  "name of the model"
  )
# Vocabulary path
flags.DEFINE_string(
  "vocab_path", 
  "./cache/dctionary/mscoco_mc3_vocab", 
  "path of the vocabulary file for the tf mRNN model"
  )
# Visual feature path
flags.DEFINE_string(
  "vf_dir", 
  "./cache/mscoco_image_features/inception_v3", 
  "directory for the visual feature"
  )
# Validation annotation files
flags.DEFINE_string(
  "anno_files_path", 
  "./datasets/ms_coco/mscoco_annos/"
  "anno_list_mscoco_crVal_m_RNN.npy",
  "Validation file annotations, multipy files should be seperated by ':'"
  )
flags.DEFINE_string(
  "results_file", 
  "vrVal_m_RNN.json", 
  "Create the images' descriptions."
  )
# Beam search size
flags.DEFINE_integer(
  "beam_size", 3, 
  "beam search size"
  )

FLAGS = flags.FLAGS


def main(unused_args):
  """Inference the images."""
  cu = CommonUtiler()
  config = ValConfig()

  # Load the r-cnn model
  decoder = mRNNDecoder(config, FLAGS.model_name, FLAGS.vocab_path, 
    gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  model_path = os.path.join(FLAGS.model_root, FLAGS.model_name)
  decoder.load_model(model_path)

    # Load the annos into memory.
  for anno_file_path in FLAGS.anno_files_path.split(':'):
    annos = np.load(anno_file_path).tolist()
    annos_file_name = anno_file_path.split("/")[-1]
    logger.info("%s loaded, %d samples" % (annos_file_name, len(annos)))

    results = []
    for anno in annos:
      feat_path = os.path.join(FLAGS.vf_dir, anno['file_path'], 
        anno['file_name'].split('.')[0] + '.txt')
      if os.path.exists(feat_path) is False:
        logger.info("%s's feature map not exists" % (anno['file_name']))
      visual_feature = np.loadtxt(feat_path)
      sens = decoder.decode(visual_feature, FLAGS.beam_size)
      result = {
        "image_id": anno['id'],
        "image_url": anno['url'],
        "image_name": anno['file_name'],
        "inference_annos": []
      }

      for sen in sens:
        re = {
          "anno": " ".join(sen['words']),
          "score": str(sen['score'])
        }
        result['inference_annos'].append(re)
      results.append(result)
      logger.info("Inference %s" % result['image_name'])

    project_dir = os.getcwd()
    results_path = os.path.join(project_dir, "results", FLAGS.results_file)
    with open(results_path, 'w') as f:
      f.write(json.dumps(results))
      logger.info("Inference finished~")


if __name__ == "__main__":
  tf.app.run()

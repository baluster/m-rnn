from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging
import tensorflow as tf


logger = logging.getLogger('helper')
logging.basicConfig(
  format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
  datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class ImageFeatureExtractor(object):
  def __init__(self, model_path):
    """Load TensorFlow CNN model."""
    assert os.path.exists(model_path), 'File does not exist %s' % model_path
    self.model_path = model_path
    # load graph
    with tf.gfile.FastGFile(os.path.join(model_path), 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')
    logger.info('Vision graph loaded from %s', model_path)
    # create a session for feature extraction
    self.session = tf.Session()
    self.writer = None
    
  def extract_features(self, image, tensor_name='pool_3:0',
                       flag_from_file=False):
    """Extract image feature from image (numpy array) or from jpeg file."""
    sess = self.session
    feat_tensor = sess.graph.get_tensor_by_name(tensor_name)
    if flag_from_file:
      # image is a path to an jpeg file
      assert os.path.exists(image), 'File does not exist %s' % image
      image_data = tf.gfile.FastGFile(image, 'rb').read()
      features = sess.run(feat_tensor, {'DecodeJpeg/contents:0': image_data})
    else:
      # image is a numpy array with image data
      image_data = image
      features = sess.run(feat_tensor, {'DecodeJpeg:0': image_data})
    
    return np.squeeze(features)
    
  def dump_graph_def(self, log_dir):
    self.writer = tf.train.SummaryWriter(log_dir, self.session.graph)


class CommonUtiler(object):
  def __init__(self):
    pass

  def split_sentence(self, sentence):
    """Tokenize a sentence. All characters are transfered to lower case."""
    # break sentence into a list of words and punctuation
    SPLIT_RE = re.compile(r'(\W+)')
    sentence = [s.strip().lower() for s in SPLIT_RE.split(sentence.strip()) \
        if len(s.strip()) > 0]
    # remove the '.' from the end of the sentence
    if sentence[-1] != '.':
      return sentence
    else:
      return sentence[:-1]

  def add_zero_fname(self, total_len, num):
    """Add zeros to file names. Useful for S3 files."""
    num_str = str(num)
    if len(num_str) > total_len:
      logger.fatal('Total length is too small to hold the number')
    return '0' * (total_len - len(num_str)) + num_str

  def truncate_list(self, l, num):
    if num == -1:
      num = len(l)
    return l[:min(len(l), num)]
    
  def create_dir_if_not_exists(self, directory):
    if not os.path.exists(directory):
      os.makedirs(directory)
      
  def load_vocabulary(self, vocab_path):
    """Initialize vocabulary from file."""
    assert os.path.exists(vocab_path), 'File does not exists %s' % vocab_path
    rev_vocab = []
    with open(vocab_path) as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
    
  def encode_sentence(self, sentence, vocab, flag_add_bos=False):
    """Encode words in a sentence with their index in vocab."""
    assert '<unk>' in vocab
    sentence_encode = []
    for word in sentence:
      if word in vocab:
        sentence_encode.append(vocab[word])
      else:
        sentence_encode.append(vocab['<unk>'])
    if flag_add_bos:
      assert '<bos>' in vocab
      sentence_encode = [vocab['<bos>']] + sentence_encode + [vocab['<bos>']]
    return sentence_encode
    
  def decode_sentence(self, sentence_encode, vocab, rev_vocab,
      flag_remove_bos=True):
    """Decode words index of a sentence to words."""
    if flag_remove_bos and sentence_encode[-1] == vocab['<bos>']:
      assert '<bos>' in vocab
      sentence = [rev_vocab[x] for x in sentence_encode[:-1]]
    else:
      sentence = [rev_vocab[x] for x in sentence_encode]
    return sentence
    
  def softmax(self, x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
  def coco_val_eval(self, pred_path, result_path):
    """Evaluate the predicted sentences on MS COCO validation."""
    sys.path.append('./external/coco-caption')
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    
    coco = COCO('./external/coco-caption/annotations/captions_val2014.json')
    cocoRes = coco.loadRes(pred_path)
    
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    
    with open(result_path, 'w') as fout:
      for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score), file=fout)
        
  def load_config(self, config_path):
    variables = {}

    # Note: the function `execfile` is not existed in the py3
    # execfile(config_path, variables)
    exec(open(config_path).read())
    return variables['config']


class Batch(object):
  """Class mRNNCocoBucketDataProvider generate Batch."""
  def __init__(self, batch_size, max_seq_len, vf_size, bos_ind):
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.vf_size = vf_size
    self.bos_ind = bos_ind
    self.empty()
      
  def empty(self):
    self.x = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
    self.y = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
    self.vf = np.zeros([self.batch_size, self.vf_size], dtype=np.float32)
    self.fg = np.zeros([self.batch_size, self.max_seq_len], dtype=np.float32)
    self.sl = np.zeros([self.batch_size], dtype=np.int32)
    self.num_feed = 0
      
  def feed_and_vomit(self, visual_features, sentence):
    i = self.num_feed
    # feed sentence
    self.x[i, 0] = self.bos_ind
    if len(sentence) > self.max_seq_len - 1:
      self.x[i, 1:] = sentence[:self.max_seq_len-1]
      self.y[i, :self.max_seq_len-1] = sentence[:self.max_seq_len-1]
      self.y[i, self.max_seq_len-1] = self.bos_ind
      self.fg[i, :] = np.ones([self.max_seq_len], dtype=np.float32)
      self.sl[i] = self.max_seq_len
    else:
      l = len(sentence)
      self.x[i, 1:l+1] = sentence
      self.y[i, :l] = sentence
      self.y[i, l] = self.bos_ind
      self.fg[i, :l+1] = np.ones([l+1], dtype=np.float32)
      self.sl[i] = l + 1
    # feed visual feature
    assert visual_features.shape[0] == self.vf_size
    self.vf[i, :] = visual_features
    self.num_feed += 1
    assert self.num_feed <= self.batch_size
    # vomit if necessary
    if self.num_feed == self.batch_size:
      return (self.x, self.y, self.vf, self.fg, self.sl)
    return None


class mRNNCocoBucketDataProvider(object):
  """mRNN TensorFlow Data Provider with Buckets on MS COCO."""
  def __init__(self, anno_files_path, vocab_path, vocab_size, vf_dir, vf_size,
      flag_shuffle=True):
    self.cu = CommonUtiler()
    self.anno_files_path = anno_files_path
    self.vocab_path = vocab_path
    self.vocab, _ = self.cu.load_vocabulary(vocab_path)
    assert len(self.vocab) == vocab_size
    assert self.vocab['<pad>'] == 0
    self.vf_dir = vf_dir
    self.vf_size = vf_size
    self.flag_shuffle = flag_shuffle
    self._load_data()
      
  def generate_batches(self, batch_size, buckets):
    """Return a list generator of mini-batches of training data."""
    # create Batches
    batches = []
    for max_seq_len in buckets:
      batches.append(
          Batch(batch_size, max_seq_len, self.vf_size, self.vocab['<bos>']))
    # shuffle if necessary
    if self.flag_shuffle:
      np.random.shuffle(self._data_pointer)
    # scan data queue
    for ind_i, ind_s in self._data_pointer:
      sentence = self._data_queue[ind_i]['sentences'][ind_s]
      visual_features = self._data_queue[ind_i]['visual_features']
      if len(sentence) >= buckets[-1]:
        feed_res = batches[-1].feed_and_vomit(visual_features, sentence)
        ind_buc = len(buckets) - 1
      else:
        for (ind_b, batch) in enumerate(batches):
          if len(sentence) < batch.max_seq_len:
            feed_res = batches[ind_b].feed_and_vomit(visual_features, sentence)
            ind_buc = ind_b
            break
      if feed_res:
        yield (ind_buc,) + feed_res
        batches[ind_buc].empty()
          
  def _load_data(self, verbose=True):
    logger.info('Loading data')
    vocab = self.vocab
    self._data_queue = []
    self._data_pointer = []
    ind_img = 0
    num_failed = 0
    for anno_file_path in self.anno_files_path:
      annos = np.load(anno_file_path).tolist()
      for (ind_a, anno) in enumerate(annos):
        data = {}
        # Load visual features
        feat_path = os.path.join(self.vf_dir, anno['file_path'],
            anno['file_name'].split('.')[0] + '.txt')
        if os.path.exists(feat_path):
          vf = np.loadtxt(feat_path)
        else:
          num_failed += 1
          continue
        data['visual_features'] = vf
        # Encode sentences
        data['sentences'] = []
        for (ind_s, sentence) in enumerate(anno['sentences']):
          sentence_encode = self.cu.encode_sentence(sentence, vocab, 
              flag_add_bos=False)
          self._data_pointer.append((ind_img, ind_s))
          data['sentences'].append(np.array(sentence_encode))
          
        self._data_queue.append(data)
        ind_img += 1
        if verbose and (ind_a + 1) % 5000 == 0:
          logger.info('Load %d/%d annotation from file %s', ind_a + 1, 
              len(annos), anno_file_path)
    
    logger.info('Load %d images, %d sentences from %d files, %d image failed', 
        len(self._data_queue), len(self._data_pointer), 
        len(self.anno_files_path), num_failed)

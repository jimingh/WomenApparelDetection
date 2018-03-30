import numpy as np
import os
import tensorflow as tf
import glob
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
  
# turn off GPU mode
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_image(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def detect_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
  
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      
      # Run inference
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
      
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]

  return output_dict

def main():
    PATH_TO_MODEL = 'Model/ssd_mobilenet_v1/frozen_inference_graph.pb'
    PATH_TO_LABELS = 'Data/label_map.pbtxt'
    NUM_CLASSES = 6
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    TEST_IMAGE_PATHS = glob.glob('InputImages/*.jpg')
    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
   
    # Array based representation of the image 
        image_np = load_image(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
        output_dict = detect_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
    plt.show()
    
main()
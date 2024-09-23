#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time
import blobconverter
from pathlib import Path

'''

Deeplabv3 person segmentation and YOLO multiclass detection
running on the same device using the Color camera in different
resolutions (according to each DL model specification).

'''

# tiny yolo v3 label texts
labelMap = [
  "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
  "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
  "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
  "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
  "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
  "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
  "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
  "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
  "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
  "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
  "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
  "teddy bear",     "hair drier", "toothbrush"
]


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", help="Shape", choices=['256','513'], default='256', type=str)
args = parser.parse_args()
segmentation_nn_shape = int(args.size)

segmentation_nn_path = blobconverter.from_zoo(name=f"deeplab_v3_mnv2_{args.size}x{args.size}", zoo_type="depthai", shaves=6)
detection_nn_path = str((Path(__file__).parent / Path('../models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())



def decode_deeplabv3p(output_tensor):
  class_colors = [[0,0,0], [0,255,0]]
  class_colors = np.asarray(class_colors, dtype=np.uint8)

  output = output_tensor.reshape(segmentation_nn_shape, segmentation_nn_shape)
  output_colors = np.take(class_colors, output, axis=0)
  return output_colors

def show_deeplabv3p(output_colors, frame):
  return cv2.addWeighted(frame,1, output_colors,0.2,0)

# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def norm_frame(frame, bbox):
  normVals = np.full(len(bbox), frame.shape[0])
  normVals[::2] = frame.shape[1]
  return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def show_yolo(name, frame, detections):
  color = (255, 0, 0)
  for detection in detections:
    bbox = norm_frame(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
    cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
  # Show the frame
  cv2.imshow(name, frame)



# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

# Define a neural network that will make segmentation predictions based on the source frames
segmentation_nn = pipeline.create(dai.node.NeuralNetwork)
segmentation_nn.setBlobPath(segmentation_nn_path)
segmentation_nn.setNumPoolFrames(4)
segmentation_nn.setNumInferenceThreads(2)
segmentation_nn.input.setBlocking(False)

# Define a neural network that will make segmentation predictions based on the source frames
detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setIouThreshold(0.1)
detection_nn.setNumClasses(80)
detection_nn.setCoordinateSize(4)
detection_nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detection_nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detection_nn.setBlobPath(detection_nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.setNumInferenceThreads(2)
detection_nn.input.setBlocking(False)

# Define camera source
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(segmentation_nn_shape, segmentation_nn_shape)
cam.setInterleaved(False)
cam.setFps(40)
cam.preview.link(segmentation_nn.input)

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(416, 416)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
manip.setKeepAspectRatio(True)
manip.out.link(detection_nn.input)
# For detection on mono camera
# cam2 = pipeline.create(dai.node.MonoCamera)
# cam2.setBoardSocket(dai.CameraBoardSocket.CAM_B)
# cam2.out.link(manip.inputImage)
cam.preview.link(manip.inputImage)

# Create outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb_seg")
xout_rgb.input.setBlocking(False)
segmentation_nn.passthrough.link(xout_rgb.input)

xout_left = pipeline.create(dai.node.XLinkOut)
xout_left.setStreamName("rgb_det")
xout_left.input.setBlocking(False)
detection_nn.passthrough.link(xout_left.input)

xout_seg_nn = pipeline.create(dai.node.XLinkOut)
xout_seg_nn.setStreamName("seg_nn")
xout_seg_nn.input.setBlocking(False)
segmentation_nn.out.link(xout_seg_nn.input)

xout_det_nn = pipeline.create(dai.node.XLinkOut)
xout_det_nn.setStreamName("det_nn")
xout_det_nn.input.setBlocking(False)
detection_nn.out.link(xout_det_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device() as device:
  cams = device.getConnectedCameras()
  device.startPipeline(pipeline)

  # Output queues will be used to get the rgb frames and nn data from the outputs defined above
  q_rgb_seg = device.getOutputQueue(name="rgb_seg", maxSize=4, blocking=False)
  q_rgb_det = device.getOutputQueue(name="rgb_det", maxSize=4, blocking=False)
  q_seg_nn = device.getOutputQueue(name="seg_nn", maxSize=4, blocking=False)
  q_det_nn = device.getOutputQueue(name="det_nn", maxSize=4, blocking=False)

  start_time = time.time()
  counter = 0
  fps = 0
  layer_info_printed = False

  while True:

    # Process Segmentation
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_rgb_seg = q_rgb_seg.get()
    in_seg_nn = q_seg_nn.get()

    if in_rgb_seg is not None:
      # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
      shape = (3, in_rgb_seg.getHeight(), in_rgb_seg.getWidth())
      frame = in_rgb_seg.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
      frame = np.ascontiguousarray(frame)

    if in_seg_nn is not None:
      # print("NN received")
      layers = in_seg_nn.getAllLayers()

      if not layer_info_printed:
        for layer_nr, layer in enumerate(layers):
          print(f"Layer {layer_nr}")
          print(f"Name: {layer.name}")
          print(f"Order: {layer.order}")
          print(f"dataType: {layer.dataType}")
          dims = layer.dims[::-1] # reverse dimensions
          print(f"dims: {dims}")
        layer_info_printed = True

      # get layer1 data
      layer1 = in_seg_nn.getLayerInt32(layers[0].name)
      # reshape to numpy array
      dims = layer.dims[::-1]
      lay1 = np.asarray(layer1, dtype=np.int32).reshape(dims)
      # decode
      output_colors = decode_deeplabv3p(lay1)

      if frame is not None:
        frame = show_deeplabv3p(output_colors, frame)
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
        cv2.imshow("rgb-seg", frame)

    counter+=1
    end_time = time.time()

    # Process Detection
    in_rgb_det = q_rgb_det.get()
    in_det_nn = q_det_nn.get()

    if in_rgb_det is not None:
      frame = in_rgb_det.getCvFrame()

    if in_det_nn is not None:
      detections = in_det_nn.detections

    if frame is not None:
      show_yolo("rgb-det", frame, detections)

    if (end_time - start_time) > 1 :
        fps = counter / (time.time() - start_time)
        counter = 0
        start_time = time.time()

    if cv2.waitKey(1) == ord('q'):
      cv2.destroyAllWindows()
      break

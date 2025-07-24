# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from synapse_msgs.msg import WarehouseShelf

import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage

import pkg_resources

import cv2
import numpy as np
import torch
import torchvision
import time
import yaml
import tflite_runtime.interpreter as tflite
from pyzbar import pyzbar

QOS_PROFILE_DEFAULT = 10

PACKAGE_NAME = 'b3rb_ros_aim_india'

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)


def xywh2xyxy(x):
    """ Converts bounding box from xywh to xyxy format. """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y

def box_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two sets of boxes.
    box1: (N, 4), box2: (M, 4)
    Returns an (N, M) IoU matrix.
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]).clamp(0) * (box[:, 3] - box[:, 1]).clamp(0)

    area1 = box_area(box1)
    area2 = box_area(box2)

    inter = (
        torch.min(box1[:, None, 2], box2[:, 2]) - torch.max(box1[:, None, 0], box2[:, 0])
    ).clamp(0) * (
        torch.min(box1[:, None, 3], box2[:, 3]) - torch.max(box1[:, None, 1], box2[:, 1])
    ).clamp(0)

    return inter / (area1[:, None] + area2 - inter + 1e-6)


def non_max_suppression(
	prediction,
	conf_thres=0.25,
	iou_thres=0.45,
	classes=None,
	agnostic=False,
	multi_label=False,
	labels=(),
	max_det=300,
	nm=0,  # number of masks
):
	"""
	Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

	Returns:
		 list of detections, on (n,6) tensor per image [xyxy, conf, cls]
	"""

	# Checks
	assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
	assert 0 <= iou_thres <= 1, f"Invalid IoU threshold value {iou_thres}, valid values are between 0.0 and 1.0"
	if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
		prediction = prediction[0]  # select only inference output

	device = prediction.device
	mps = "mps" in device.type  # Apple MPS
	if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
		prediction = prediction.cpu()
	bs = prediction.shape[0]  # batch size
	nc = prediction.shape[2] - nm - 5  # number of classes
	xc = prediction[..., 4] > conf_thres  # candidates

	# Settings
	max_wh = 7680  # (pixels) maximum box width and height
	max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
	time_limit = 0.5 + 0.05 * bs  # seconds to quit after
	redundant = True  # require redundant detections
	multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
	merge = False  # use merge-NMS

	t = time.time()
	mi = 5 + nc  # mask start index
	output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
	for xi, x in enumerate(prediction):  # image index, image inference
		# Apply constraints
		x = x[xc[xi]]  # confidence

		# Cat apriori labels if autolabelling
		if labels and len(labels[xi]):
			lb = labels[xi]
			v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
			v[:, :4] = lb[:, 1:5]  # box
			v[:, 4] = 1.0  # conf
			v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
			x = torch.cat((x, v), 0)

		# If none remain process next image
		if not x.shape[0]:
			continue

		# Compute conf
		x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

		# Box/Mask
		box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
		mask = x[:, mi:]  # zero columns if no masks

		# Detections matrix nx6 (xyxy, conf, cls)
		if multi_label:
			i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
			x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
		else:  # best class only
			conf, j = x[:, 5:mi].max(1, keepdim=True)
			x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

		# Filter by class
		if classes is not None:
			x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

		# Check shape
		n = x.shape[0]  # number of boxes
		if not n:  # no boxes
			continue
		x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

		# Batched NMS
		c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
		boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
		i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
		i = i[:max_det]  # limit detections
		if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
			# update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
			iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
			weights = iou * scores[None]  # box weights
			x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
			if redundant:
				i = i[iou.sum(1) > 1]  # require redundancy

		output[xi] = x[i]
		if mps:
			output[xi] = output[xi].to(device)
		if (time.time() - t) > time_limit:
			break  # time limit exceeded

	return output


class ObjectRecognizer(Node):
    def __init__(self):
        super().__init__('object_recognizer')

        model_path = '/home/pavan/cognipilot/cranium/src/NXP_AIM_INDIA_2025/resource/yolov5n-int8.tflite'
        label_path = '/home/pavan/cognipilot/cranium/src/NXP_AIM_INDIA_2025/resource/coco.yaml'

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.img_size = self.input_details[0]['shape'][1]

        with open(label_path, 'r') as f:
            self.label_names = yaml.load(f, Loader=yaml.FullLoader)['names']

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        self.publisher_shelf_objects = self.create_publisher(
            WarehouseShelf,
            '/shelf_objects',
            QOS_PROFILE_DEFAULT)


        self.publisher_debug_image = self.create_publisher(
            CompressedImage,
            '/debug_images/object_recog',
            QOS_PROFILE_DEFAULT)

        self.last_qr_data = None

        # ✅ Allowed objects list
        self.allowed_objects = {
            "horse", "clock", "zebra", "potted plant", "cup", "banana", "car", "teddy bear"
        }

    def camera_image_callback(self, message):
        # self.get_logger().info("📷 [ObjectRecognizer] Got image")
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            original_h, original_w, _ = image.shape

            # --- QR CODE ---
            barcodes = pyzbar.decode(image)
            for barcode in barcodes:
                qr_data = barcode.data.decode("utf-8")
                self.last_qr_data = qr_data
                x, y, w, h = barcode.rect
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # BLUE
                cv2.putText(image, f"QR: {qr_data}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                self.get_logger().info(f"📦 QR Detected: {qr_data}")

            # --- OBJECT DETECTION ---
            resized_img = cv2.resize(image, (self.img_size, self.img_size))
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            img = rgb_img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            input_tensor = self.input_details[0]
            is_quant = input_tensor["dtype"] == np.uint8
            if is_quant:
                scale, zero_point = input_tensor["quantization"]
                img = (img / scale + zero_point).astype(np.uint8)
            self.interpreter.set_tensor(input_tensor["index"], img)

            start = time.time()
            self.interpreter.invoke()
            self.get_logger().info(f"Inference time: {(time.time() - start) * 1000:.1f} ms")

            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            if is_quant:
                out_scale, out_zp = self.output_details[0]["quantization"]
                output = (output.astype(np.float32) - out_zp) * out_scale

            output[..., :4] *= [self.img_size] * 4
            pred = torch.tensor(output)
            detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

            counts = {}
            if detections is not None and len(detections) > 0:
                for *xyxy, conf, cls in detections:
                    label_raw = self.label_names[int(cls)]
                    label = "potted plant" if label_raw == "plant" else label_raw

                    start = (int(xyxy[0]), int(xyxy[1]))
                    end = (int(xyxy[2]), int(xyxy[3]))

                    # ✅ Green for valid objects
                    if label in self.allowed_objects:
                        color = (0, 255, 0)  # Green
                        counts[label] = counts.get(label, 0) + 1
                    else:
                        color = (0, 0, 255)  # Red

                    cv2.rectangle(resized_img, start, end, color, 2)
                    cv2.putText(resized_img, f"{label} {float(conf):.2f}", (start[0], start[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ✅ Publish shelf data with full QR info
            if counts:
                msg = WarehouseShelf()
                msg.object_name = list(counts.keys())
                msg.object_count = list(counts.values())
                msg.qr_decoded = self.last_qr_data or ""
                self.publisher_shelf_objects.publish(msg)

            debug_img = cv2.resize(resized_img, (original_w, original_h))
            self.publish_debug_image(self.publisher_debug_image, debug_img)

        except Exception as e:
            self.get_logger().error(f"[ObjectRecognizer] Error: {e}")

    def publish_debug_image(self, publisher, image):
        msg = CompressedImage()
        msg.format = "jpeg"
        _, encoded = cv2.imencode('.jpg', image)
        msg.data = encoded.tobytes()
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
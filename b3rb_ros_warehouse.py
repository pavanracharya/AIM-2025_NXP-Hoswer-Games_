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
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading

from pyzbar import pyzbar

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped, Quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf_transformations 

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

import tkinter as tk
from tkinter import ttk
from tf2_ros import Buffer, TransformListener


QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True


class WindowProgressTable:
	def __init__(self, root, shelf_count):
		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)

		self.row_count = 2
		self.col_count = shelf_count

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
					      relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)

	def change_box_color(self, row, col, color):
		self.boxes[row][col].config(bg=color)

	def change_box_text(self, row, col, text):
		self.boxes[row][col].delete(1.0, tk.END)
		self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()

# -------- QR Curtain Logic Helper --------
class ShelfQRManager:
    def __init__(self):
        self.qr_sequence = []  # List of QR codes in correct order
        self.published_qrs = []  # List of QR codes already published

    def add_detected_qr(self, qr_str):
        """Register a new QR if it’s not already seen."""
        if qr_str not in self.qr_sequence:
            self.qr_sequence.append(qr_str)

    def is_ready_to_publish(self, qr_str):
        """Return True if all QRs before this one were already published."""
        if qr_str not in self.qr_sequence:
            return False  # Unknown shelf QR

        index = self.qr_sequence.index(qr_str)
        required = self.qr_sequence[:index]
        return all(qr in self.published_qrs for qr in required)

    def mark_published(self, qr_str):
        if qr_str not in self.published_qrs:
            self.published_qrs.append(qr_str)


class WarehouseExplore(Node):
	""" Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')
		self.last_shelf_print_time = time.time()
		# self.qr_manager = ShelfQRManager()
		self.qr_col_map = {}
		self.table_col_count = 0
		self.initial_angle_sent = False  # in __init__
		self.declare_parameter('initial_angle', 0.0)
		self.declare_parameter('x', 0.0)
		self.declare_parameter('y', 0.0)
		self.declare_parameter('shelf_count', 1)
		self.shelf_count = self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = self.get_parameter('initial_angle').get_parameter_value().double_value
		self.start_x = self.get_parameter('x').get_parameter_value().double_value
		self.start_y = self.get_parameter('y').get_parameter_value().double_value
		self._frame_id = "map"
		self.goal_completed = True
		self.goal_handle_curr = None
		self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		self.qr_view_sent = False
		self.initial_angle_used = False
		self.current_warehouse = "WH1"
		# self.warehouse_shelves = {
		# 	"WH1": [(1,-2.893, 2.905, 0.785), (2, 2.824, -2.854, 1.396)],
		# 	"WH2": [(1, 4.822, -4.818, 0.121), (2, 1.888, -0.922, 0.907), (3, 1.020, 3.773, -0.549), (4, 6.692, 5.999, -0.777)],
		# 	"WH3": [(1, 1.927, 1.928, 2.330), (2, 3.371, 4.863, -0.733), (3, 2.019, 8.919, -0.025)],
		# 	"WH4": [(1, -2.344, -1.718, 1.541), (2, 3.908, -5.821, 0.508), (3, -0.030, -4.930, 0.000),
		# 			(4, 2.956, -1.966, 0.785), (5, 1.951, 3.647, 0.003), (6, 5.045, 3.651, 0.000)],
		# }
 # flag to ensure shelf nav only starts after initial_angle
		self.qr_goal_sent = False
		self.obj_view_sent = False
		self.timer_retry_initialized = False
		self.shelf_retry_timer = None
		self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
		self.nav_to_pose_client.wait_for_server()
		# Shelf exploration state
		self.exploration_started = False
		self.shelf_detected = False
		self.forward_check_distance = 2.0  # total distance to travel (meters)
		self.forward_step = 0.3            # each forward step (meters)
		self.forward_progress = 0.0
		self.initial_pose = None

		# Example shelf-1 reference for WH1 (tweak per warehouse)



		# Start a timer to control step movement
		self.timer = self.create_timer(1.0, self.exploration_timer_callback)  # 1 Hz



		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')

		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)


		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/amcl_pose',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)
		




		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		self.full_map_explored_count = 0

		# --- QR Code Data ---
		self.qr_code_str = "Empty"
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0

		# --- Shelf Data ---
		self.shelf_objects_curr = WarehouseShelf()



	def pose_callback(self, message):
		self.pose_curr = message

		self.get_logger().info(f"[DEBUG] Pose received: {message.pose.pose.position.x}, {message.pose.pose.position.y}")
		self.get_logger().info(f"[DEBUG] initial_angle={self.initial_angle}, sent={self.initial_angle_sent}")

		# Already sent or angle is zero, skip
		if self.initial_angle_sent or self.initial_angle == 0.0:
			return

		try:
			if not self.tf_buffer.can_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
				self.get_logger().warn("⚠️ TF map → base_link not available yet. Waiting...")
				return

			trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
			x = trans.transform.translation.x
			y = trans.transform.translation.y
			self.get_logger().info(f"[DEBUG] TF transform: x={x}, y={y}")

			yaw_rad = math.radians(self.initial_angle)

			# Step forward by 0.5m in direction of initial_angle
			step_dist = 0.5
			x_fwd = x + step_dist * math.cos(yaw_rad)
			y_fwd = y + step_dist * math.sin(yaw_rad)

			goal = PoseStamped()
			goal.header.frame_id = "map"
			goal.header.stamp = self.get_clock().now().to_msg()
			goal.pose.position.x = x_fwd
			goal.pose.position.y = y_fwd
			goal.pose.position.z = 0.0

			quat = tf_transformations.quaternion_from_euler(0, 0, yaw_rad)
			goal.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])


			self.initial_angle_sent = True
			self.step_forward_along_angle()  # Start straight movement after initial angle rotation
			self.get_logger().info(f"🎯 Rotated and stepped forward in initial_angle: {self.initial_angle}°")
			self.exploration_started = True
			self.forward_progress = 0.0
			self.initial_pose = (x, y)

		except Exception as e:
			self.get_logger().warn(f"⚠️ TF2 lookup failed: {str(e)}")

	def exploration_timer_callback(self):
		if not self.exploration_started:
			return

		if self.forward_progress >= self.forward_check_distance:
			self.get_logger().info("🛑 Max forward distance reached.")
			self.exploration_started = False
			return

		if self.shelf_detected:
			self.get_logger().info("✅ Shelf detected by camera. Halting forward motion.")
			self.exploration_started = False
			self.save_current_pose()  # You should implement this
			return

		try:
			if not self.tf_buffer.can_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
				self.get_logger().warn("⚠️ TF map → base_link not available for forward step.")
				return

			trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
			yaw = math.radians(self.initial_angle)

			step_x = self.forward_step * math.cos(yaw)
			step_y = self.forward_step * math.sin(yaw)

			goal = PoseStamped()
			goal.header.frame_id = "map"
			goal.header.stamp = self.get_clock().now().to_msg()
			goal.pose.position.x = self.initial_pose[0] + self.forward_progress * math.cos(yaw) + step_x
			goal.pose.position.y = self.initial_pose[1] + self.forward_progress * math.sin(yaw) + step_y
			goal.pose.position.z = 0.0

			quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
			goal.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])


			self.forward_progress += self.forward_step

			self.get_logger().info(f"🚶 Step forward: {self.forward_progress:.2f} m")

		except Exception as e:
			self.get_logger().warn(f"❌ Exploration TF error: {str(e)}")


	def save_current_pose(self):
		try:
			trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
			x = trans.transform.translation.x
			y = trans.transform.translation.y
			self.get_logger().info(f"💾 Shelf pose saved at x={x:.2f}, y={y:.2f}")
			# You can later generate QR/object view goals from here
		except Exception as e:
			self.get_logger().warn(f"❌ Could not save pose: {str(e)}")


	def move_forward_using_nav2(self, distance=1.5):
		"""
		Moves the robot forward by 'distance' meters using Nav2, based on initial_angle.
		Avoids obstacles and shelf legs using Nav2 planner.
		"""
		try:
			# Wait for TF between map → base_link
			if not self.tf_buffer.can_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
				self.get_logger().warn("⚠️ TF map → base_link not available. Waiting...")
				return

			# Get current position from TF
			trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
			x = trans.transform.translation.x
			y = trans.transform.translation.y
			self.get_logger().info(f"[DEBUG] TF current position: x={x}, y={y}")

			# Compute forward target
			yaw_rad = math.radians(self.initial_angle)
			x_offset = x + distance * math.cos(yaw_rad)
			y_offset = y + distance * math.sin(yaw_rad)

			# Create PoseStamped goal
			goal = PoseStamped()
			goal.header.frame_id = "map"
			goal.header.stamp = self.get_clock().now().to_msg()
			goal.pose.position.x = x_offset
			goal.pose.position.y = y_offset
			goal.pose.position.z = 0.0

			quat = tf_transformations.quaternion_from_euler(0, 0, yaw_rad)
			goal.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

			self.get_logger().info(f"🧭 Sending forward Nav2 goal: x={x_offset:.2f}, y={y_offset:.2f}, yaw={self.initial_angle}°")


		except Exception as e:
			self.get_logger().warn(f"⚠️ move_forward_using_nav2 failed: {str(e)}")

	def step_forward_until_shelf(self):
		if not self.pose_curr:
			self.get_logger().warn("⚠️ Current pose not available.")
			return

		# Set your reference shelf coordinate for stopping logic
		ref_x, ref_y = self.first_shelf_ref_coords  # e.g., (2.8, -2.8)

		# Current pose
		curr_x = self.pose_curr.pose.pose.position.x
		curr_y = self.pose_curr.pose.pose.position.y

		# Check angle quadrant and stop if shelf reached
		angle = self.initial_angle
		should_stop = False

		if 45 <= angle < 135:  # Facing +y direction
			if curr_y >= ref_y:
				should_stop = True
		elif 135 <= angle < 225:  # Facing -x direction
			if curr_x <= ref_x:
				should_stop = True
		elif 225 <= angle < 315:  # Facing -y direction
			if curr_y <= ref_y:
				should_stop = True
		else:  # Facing +x direction
			if curr_x >= ref_x:
				should_stop = True

		if should_stop:
			self.get_logger().info("🛑 Shelf reference reached. Stopping forward steps.")

			self.saved_shelf_pose = self.pose_curr
			self.get_logger().info(f"📌 Shelf pose saved: x={curr_x:.2f}, y={curr_y:.2f}")
			# You can call next step here: self.move_to_qr_view() or so
			return

		# Send next step forward goal
		step_distance = 0.3  # meters
		yaw_rad = math.radians(self.initial_angle)
		next_x = curr_x + step_distance * math.cos(yaw_rad)
		next_y = curr_y + step_distance * math.sin(yaw_rad)

		self.get_logger().info(f"🚶 Step forward to: ({next_x:.2f}, {next_y:.2f})")
		goal = self.create_goal_from_world_coord(next_x, next_y, yaw_rad)
		self.send_goal_to_nav2_internal(goal)

	# def move_to_qr_view(self, lateral_offset=0.5, angle_offset_deg=90.0):
	# 	self.logger.info("📤 Calling move_to_qr_view()...")

	# 	# Get shelf-1 pose
	# 	shelf = self.warehouse_shelves[self.current_warehouse][0]
	# 	_, x, y, yaw_shelf = shelf

	# 	# Compute new yaw (rotate left or right to side view)
	# 	yaw_qr = yaw_shelf + math.radians(angle_offset_deg)

	# 	# Compute side offset
	# 	qr_x = x + lateral_offset * math.cos(yaw_qr)
	# 	qr_y = y + lateral_offset * math.sin(yaw_qr)

	# 	# Create goal pose
	# 	goal = self.create_goal_from_world_coord(qr_x, qr_y, yaw_qr)
	# 	self.logger.info(f"🧭 Sending QR goal to x={qr_x:.2f}, y={qr_y:.2f}, yaw={yaw_qr:.2f}")
	# 	self.send_goal_from_world_pose(goal)

	# def _get_yaw_from_pose(self, orientation: Quaternion) -> float:
	# 	quat = [orientation.x, orientation.y, orientation.z, orientation.w]
	# 	_, _, yaw = tf_transformations.euler_from_quaternion(quat)
	# 	return yaw


	def simple_map_callback(self, message):
		"""Callback function to handle simple map updates.

		Args:
			message: ROS2 message containing the simple map data.

		Returns:
			None
		"""
		self.simple_map_curr = message
		map_info = self.simple_map_curr.info
		self.world_center = self.get_world_coord_from_map_coord(
			map_info.width / 2, map_info.height / 2, map_info
		)

	def global_map_callback(self, message):
		return
		"""Callback function to handle global map updates."""
		self.global_map_curr = message
		pass
		if not self.goal_completed:
			return

		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_array = np.array(self.global_map_curr.data).reshape((height, width))

		frontiers = self.get_frontiers_for_space_exploration(map_array)

		map_info = self.global_map_curr.info
		if frontiers:
			closest_frontier = None
			min_distance_curr = float('inf')

			for fy, fx in frontiers:
				fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy, map_info)
				distance = euclidean((fx_world, fy_world), self.buggy_center)
				if (distance < min_distance_curr and
					distance <= self.max_step_dist_world_meters and
					distance >= self.min_step_dist_world_meters):
					min_distance_curr = distance
					closest_frontier = (fy, fx)

			if closest_frontier:
				fy, fx = closest_frontier
				goal = self.create_goal_from_map_coord(fx, fy, map_info)
				self.send_goal_from_world_pose(goal)
				print("🔁 Sending goal for automatic space exploration.")
				return
			else:
				# Adjust exploration distance thresholds
				self.max_step_dist_world_meters += 2.0
				self.min_step_dist_world_meters = max(0.25, self.min_step_dist_world_meters - 1.0)

			self.full_map_explored_count = 0
		else:
			self.full_map_explored_count += 1
			print(f"🔚 No frontiers found. Count: {self.full_map_explored_count}")


			
	def get_frontiers_for_space_exploration(self, map_array):
		"""Identifies frontiers for space exploration.

		Args:
			map_array: 2D numpy array representing the map.

		Returns:
			frontiers: List of tuples representing frontier coordinates.
		"""
		frontiers = []
		for y in range(1, map_array.shape[0] - 1):
			for x in range(1, map_array.shape[1] - 1):
				if map_array[y, x] == -1:  # Unknown space and not visited.
					neighbors_complete = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
						(y - 1, x - 1),
						(y + 1, x - 1),
						(y - 1, x + 1),
						(y + 1, x + 1)
					]

					near_obstacle = False
					for ny, nx in neighbors_complete:
						if map_array[ny, nx] > 0:  # Obstacles.
							near_obstacle = True
							break
					if near_obstacle:
						continue

					neighbors_cardinal = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
					]

					for ny, nx in neighbors_cardinal:
						if map_array[ny, nx] == 0:  # Free space.
							frontiers.append((ny, nx))
							break

		return frontiers



	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)

	def camera_image_callback(self, message):
		"""Callback for processing compressed camera images and detecting QR codes."""
		try:
			# Convert compressed image to OpenCV image
			np_arr = np.frombuffer(message.data, np.uint8)
			image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

			# Detect QR codes using pyzbar
			barcodes = pyzbar.decode(image)

			self.get_logger().info(f"Found {len(barcodes)} QR code(s)")

			decoded_text = ""
			for barcode in barcodes:
				barcode_data = barcode.data.decode("utf-8")
				barcode_type = barcode.type
				x, y, w, h = barcode.rect

				self.get_logger().info(f"QR Detected: {barcode_type} - {barcode_data}")
				decoded_text = barcode_data  # save latest QR decoded (use first one or overwrite)

				# Draw bounding box in blue
				cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
				cv2.putText(image, barcode_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (255, 0, 0), 2)

			# Publish to debug image topic
			self.publish_debug_image(self.publisher_qr_decode, image)

			# Publish to /shelf_objects (only QR)
			if decoded_text:
				msg = WarehouseShelf()
				msg.qr_decoded = decoded_text
				msg.object_name = []
				msg.object_count = []
				# self.publisher_shelf_objects.publish(msg)

		except Exception as e:
			self.get_logger().error(f"Error in QR detection: {e}")

	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates.

		Args:
			message: ROS2 message containing cerebri status.

		Returns:
			None
		"""
		if message.mode == 3 and message.arming == 2:
			self.armed = True
		else:
			# Initialize and arm the CMD_VEL mode.
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			# self.publisher_joy.publish(msg)

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass

	def shelf_objects_callback(self, message):
		self.shelf_objects_curr = message
		qr_str = message.qr_decoded.strip()
		if not qr_str:
			return

		# Assign a GUI column to this shelf if first time
		if not hasattr(self, 'qr_col_map'):
			self.qr_col_map = {}
			self.table_col_count = 0

		if qr_str not in self.qr_col_map:
			self.qr_col_map[qr_str] = self.table_col_count
			self.table_col_count += 1

		col = self.qr_col_map[qr_str]

		# Track object names and counts as-is (no filtering, no merging)
		final_msg = WarehouseShelf()
		final_msg.qr_decoded = qr_str
		final_msg.object_name = message.object_name
		final_msg.object_count = message.object_count

		# Publish directly to /shelf_data
		self.publisher_shelf_data.publish(final_msg)
		print(f"✅ Published: {qr_str} with {len(final_msg.object_name)} objects")

		# Update GUI immediately
		if PROGRESS_TABLE_GUI:
			obj_str = "\n".join(f"{n}: {c}" for n, c in zip(final_msg.object_name, final_msg.object_count))
			box_app.change_box_text(0, col, obj_str)
			box_app.change_box_color(0, col, "cyan")
			box_app.change_box_text(1, col, qr_str)
			box_app.change_box_color(1, col, "yellow")


	# def shelf_objects_callback(self, message):
	# 	"""Callback function to handle shelf objects updates.

	# 	Args:
	# 		message: ROS2 message containing shelf objects data.

	# 	Returns:
	# 		None
	# 	"""
	# 	self.shelf_objects_curr = message
	# 	# Process the shelf objects as needed.

	# 	# How to send WarehouseShelf messages for evaluation.

	# 	# * Example for sending WarehouseShelf messages for evaluation.
	# 	# shelf_data_message = WarehouseShelf()

	# 	# shelf_data_message.object_name = ["car", "clock"]
	# 	# shelf_data_message.object_count = [1, 2]
	# 	# shelf_data_message.qr_decoded = "test qr string"

	# 	# self.publisher_shelf_data.publish(shelf_data_message)

	# 	# * Alternatively, you may store the QR for current shelf as self.qr_code_str.
	# 	# 	Then, add it as self.shelf_objects_curr.qr_decoded = self.qr_code_str
	# 	# 	Then, publish as self.publisher_shelf_data.publish(self.shelf_objects_curr)


	# 	# Optional code for populating TABLE GUI with detected objects and QR data.

	# 	if PROGRESS_TABLE_GUI:
	# 		shelf = self.shelf_objects_curr
	# 		obj_str = ""
	# 		for name, count in zip(shelf.object_name, shelf.object_count):
	# 			obj_str += f"{name}: {count}\n"

	# 		box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
	# 		box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
	# 		self.table_row_count += 1

	# 		box_app.change_box_text(self.table_row_count, self.table_col_count, self.qr_code_str)
	# 		box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
	# 		self.table_row_count = 0
	# 		self.table_col_count += 1
		


	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)



	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False


	def step_forward_until_detection(self, max_steps=7, step_distance=0.3):
		for _ in range(max_steps):
			if self.qr_detected or self.object_detected:
				self.get_logger().info("Shelf detected, stopping step-forward")
				self.shelf_front_pose = self.get_robot_pose()
				break

			self.get_logger().info("No shelf detected, stepping forward...")
			self.send_step_goal(distance=step_distance)
			rclpy.spin_once(self, timeout_sec=2.0)

	def send_step_goal(self, distance):
		pose = self.get_robot_pose()
		new_x = pose.position.x + distance * math.cos(self.current_yaw)
		new_y = pose.position.y + distance * math.sin(self.current_yaw)
		yaw = self.current_yaw


	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)
	def goal_result_callback(self, future):
		status = future.result().status

		if status == GoalStatus.STATUS_SUCCEEDED:
			self.logger.info("🎯 Goal completed successfully.")
		else:
			self.logger.warn(f"❌ Goal failed with status: {status}")

		self.goal_completed = True
		self.goal_handle_curr = None

		# Only run this loop during first shelf search
		if not self.initial_angle_sent:
			self.get_logger().info("🔁 Continuing toward shelf using step_forward_until_shelf()...")
			self.step_forward_until_shelf()


	def _retry_shelf_nav(self):
		try:
			# Wait for AMCL


			# Wait for costmap transform
			if self.get_robot_pose_from_costmap() is None:
				raise ValueError("Costmap pose not available")

			# If both are ready: cancel timer, send goal
			self.shelf_retry_timer.cancel()
			self.logger.info("✅ AMCL & costmap ready. Sending shelf-1 goal...")

			shelf = self.warehouse_shelves[self.current_warehouse][0]
			_, x, y, yaw = shelf
			self.logger.info(f"📍 Navigating to shelf-1 at (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f})")

			goal = self.create_goal_from_world_coord(x, y, yaw)


		except Exception as e:
			self.logger.warn(f"⏳ Shelf nav retry: {e}")



	def send_goal_to_nav2_internal(self, goal_pose):
		from nav2_msgs.action import NavigateToPose

		nav_goal = NavigateToPose.Goal()
		nav_goal.pose = goal_pose

		self.logger.info(
			f"🧭 Sending goal to Nav2: x={goal_pose.pose.position.x:.2f}, y={goal_pose.pose.position.y:.2f}"
		)

		future = self.nav_to_pose_client.send_goal_async(
			nav_goal,
			feedback_callback=self.goal_feedback_callback  # optional
		)
		future.add_done_callback(self.goal_response_callback)


	def offset_goal_from_shelf(self, x: float, y: float, yaw: float, offset: float = 0.4):
		"""
		Offsets the goal away from the shelf by `offset` meters along reverse yaw.
		Ensures the robot stops slightly before the shelf, not colliding into it.
		"""
		x_offset = x - offset * math.cos(yaw)
		y_offset = y - offset * math.sin(yaw)
		return x_offset, y_offset, yaw


	def get_robot_pose_from_costmap(self):
		if self.pose_curr is None:
			return None
		return self.pose_curr.pose.pose  # crude but safe fallback





	def get_amcl_pose(self):
		try:
			msg = self.node.get_parameter_or("amcl_pose", None)
			if msg is not None and hasattr(msg, 'pose'):
				return msg
		except:
			return None



	def _get_yaw_from_pose(self, orientation: Quaternion) -> float:
		"""
		Converts a Quaternion into yaw (rotation around Z axis).

		Args:
			orientation (geometry_msgs.msg.Quaternion): Quaternion orientation

		Returns:
			float: Yaw in radians
		"""
		quat = [orientation.x, orientation.y, orientation.z, orientation.w]
		_, _, yaw = tf_transformations.euler_from_quaternion(quat)
		return yaw

	def goal_response_callback(self, future):
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn("❌ Goal rejected.")
			self.goal_completed = True
			self.goal_handle_curr = None
		else:
			self.logger.info("✅ Goal accepted.")
			self.goal_completed = False
			self.goal_handle_curr = goal_handle

			result_future = goal_handle.get_result_async()
			result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
			self.cancel_current_goal()  # Unblock by discarding the current goal.

	def send_goal_from_world_pose(self, goal_pose: PoseStamped):
		if self.goal_handle_curr is not None and not self.goal_completed:
			self.get_logger().warn("A goal is already active, skipping new goal.")
			self.get_logger().warn(f"❌ Skipping new goal. goal_completed={self.goal_completed}")
			return False

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.get_logger().error("NavigateToPose action server not available!")
			return False

		nav_goal = NavigateToPose.Goal()
		nav_goal.pose = goal_pose
		nav_goal.behavior_tree = ""  # default BT

		self.get_logger().info(
			f"🧭 Sending goal to x={goal_pose.pose.position.x:.2f}, "
			f"y={goal_pose.pose.position.y:.2f}, frame={goal_pose.header.frame_id}"
		)
		self.get_logger().info(f"[DEBUG] Goal being sent:\n{nav_goal.pose}")

		send_goal_future = self.action_client.send_goal_async(
			nav_goal,
			feedback_callback=self.goal_feedback_callback
		)
		send_goal_future.add_done_callback(self.goal_response_callback)

		self.goal_completed = False
		return True



	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
									yaw: Optional[float] = None) -> PoseStamped:
		"""
		Creates a goal PoseStamped from world coordinates (map frame).
		This works even if current robot pose is unavailable, but includes auto-yaw fallback if needed.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id  # usually "map"

		# Set position
		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y
		goal_pose.pose.position.z = 0.0

		# Determine yaw (if not provided)
		if yaw is None and self.pose_curr is not None:
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0  # fallback if pose_curr not available

		# Set orientation using helper
		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		self.get_logger().info(
			f"[DEBUG] Goal created: x={world_x:.2f}, y={world_y:.2f}, yaw={yaw:.2f} rad"
		)
		return goal_pose
	def get_shelf_side_goal(self, shelf_x, shelf_y, shelf_yaw, offset=0.5):
		# Offset backward from shelf center along yaw direction
		goal_x = shelf_x - offset * math.cos(shelf_yaw)
		goal_y = shelf_y - offset * math.sin(shelf_yaw)
		return self.create_goal_from_world_coord(goal_x, goal_y, shelf_yaw)

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
				       yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
	rclpy.init(args=args)

	warehouse_explore = WarehouseExplore()

	if PROGRESS_TABLE_GUI:
		gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
		gui_thread.start()

	rclpy.spin(warehouse_explore)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	warehouse_explore.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()

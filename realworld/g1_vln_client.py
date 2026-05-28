import rclpy
import sys
import threading
import PIL.Image as PIL_Image
import io
import json
import os
import requests
import time
import numpy as np
import math

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# unitree related
from unitree_api.msg import Request, RequestHeader, RequestIdentity
from unitree_go.msg import SportModeState

# user-specific
from pid_controller import *
from utils import ReadWriteLock
# global variable
policy_init = True
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
manager = None
REPLAN_POS_THRESH = 0.1
REPLAN_YAW_THRESH = math.radians(15.0)
GOAL_TIMEOUT_SEC = 8.0
run_time = time.strftime("%Y%m%d_%H%M%S")
video_dir = os.path.join("video", run_time)
save_frame_idx = 0

rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()


def save_inference_image(rgb_image):
    global save_frame_idx
    os.makedirs(video_dir, exist_ok=True)
    save_frame_idx += 1
    image = PIL_Image.fromarray(rgb_image)
    save_path = os.path.join(video_dir, f"{save_frame_idx:06d}.jpg")
    image.save(save_path, format="JPEG", quality=95)
    print(f"[VIDEO] Saved inference image: {save_path}")
    
def eval_vln(image, depth, camera_pose, instruction, url='http://192.168.0.170:5801/eval_vln'):
    global policy_init
    image = PIL_Image.fromarray(image)
    # image = image.resize((384, 384))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='jpeg')
    image_bytes.seek(0)
    
    data = {"reset":policy_init}
    json_data = json.dumps(data)
    policy_init = False
    
    files = {'image': ('rgb_image', image_bytes, 'image/jpg')}
    start = time.time()
    response = requests.post(url, files=files,data={'json': json_data} , timeout=150)
    print(f"total time(delay + policy): {time.time() - start}")
    print(response.text)
    
    action = json.loads(response.text)['action']
    return action

def control_thread():
    last_debug_time = 0.0
    while True:
        odom_rw_lock.acquire_read()
        homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
        vel = manager.vel.copy() if manager.vel is not None else None
        homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None
        odom_rw_lock.release_read()

        e_p, e_r = 0.0, 0.0
        if homo_odom is not None and vel is not None and homo_goal is not None:
            v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
            if v < 0.0:
                v = 0.0
            manager.move(v, 0, w)
            now = time.time()
            if now - last_debug_time > 1.0:
                print(f"[CONTROL] v={v:.3f}, w={w:.3f}, e_p={e_p:.3f}, e_yaw={math.degrees(e_r):.1f}deg")
                last_debug_time = now
        goal_age = time.time() - manager.last_goal_update_time if manager is not None else 0.0
        if abs(e_p) < REPLAN_POS_THRESH and abs(e_r) < REPLAN_YAW_THRESH:
            print(f"[CONTROL] goal reached, trigger replan: e_p={e_p:.3f}, e_yaw={math.degrees(e_r):.1f}deg")
            manager.trigger_replan()
        elif abs(e_p) < REPLAN_POS_THRESH and goal_age > GOAL_TIMEOUT_SEC:
            print(f"[CONTROL] yaw goal timeout, trigger replan: e_p={e_p:.3f}, e_yaw={math.degrees(e_r):.1f}deg")
            manager.trigger_replan()
        time.sleep(0.1)


def planning_thread():
    while True:
        if not manager.should_plan:
            continue
        
        print(f"planning_thread running")
        rgb_depth_rw_lock.acquire_read()
        rgb_image = manager.rgb_image
        rgb_depth_rw_lock.release_read()

        odom_rw_lock.acquire_read()
        request_cnt = manager.request_cnt
        odom_rw_lock.release_read()
        if rgb_image is None:
            time.sleep(0.1)
            continue

        save_inference_image(rgb_image)
        actions = eval_vln(rgb_image, None, None, None)
        print(f"[PLAN] actions={actions}")
        odom_rw_lock.acquire_write()
        manager.should_plan = False
        manager.request_cnt += 1
        manager.incremental_change_goal(actions)
        odom_rw_lock.release_write()
        time.sleep(0.1)


class G1VlnManager(Node):
    def __init__(self):
        super().__init__('g1_vln_manager')

        # subsucriber
        rgb_sub = Subscriber(self, Image, "/camera/color/image_raw")
        depth_sub = Subscriber(self, Image, "/camera/aligned_depth_to_color/image_raw")
        self.syncronizer = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_callback)

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.odom_sub = self.create_subscription(SportModeState, "/lf/odommodestate", self.odom_callback, qos_profile)

        # publisher
        self.control_pub = self.create_publisher(Request, '/api/sport/request', 5)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 5)

        # class member variable
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.rgb_time = 0.0
        self.depth_time = 0.0
        self.homo_goal = None
        self.homo_odom = None
        self.vel = None
        self.odom = None
        
        self.request_cnt = 0
        self.odom_cnt = 0
        
        self.should_plan = False
        self.last_plan_time = 0.0
        self.last_goal_update_time = time.time()
    def rgb_depth_callback(self, rgb_msg, depth_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0

        rgb_depth_rw_lock.acquire_write()
        self.rgb_image = raw_image
        self.depth_image = raw_depth / 1000.0
        self.depth_image[np.where(self.depth_image < 0)] = 0
        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        self.depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1.0e9
        rgb_depth_rw_lock.release_write()

    def odom_callback(self, msg):
        self.odom_cnt += 1

        odom_rw_lock.acquire_write()
        x, y = msg.position[0], msg.position[1]
        vx = msg.velocity[0]
        yaw = msg.imu_state.rpy[2]
        vyaw = msg.yaw_speed

        self.odom = [x, y, yaw]
        R0 = np.array([[np.cos(yaw), -np.sin(yaw)],
                       [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [x, y]
        self.vel = [vx, vyaw]
        
        if self.odom_cnt == 1:
            # fisrst odom
            self.homo_goal = self.homo_odom.copy()
        odom_rw_lock.release_write()

    def trigger_replan(self):
        self.should_plan = True

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_odom.copy()
        
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle),  math.cos(angle), 0],
                    [0,                0,               1]
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle),  math.cos(angle), 0],
                    [0,                0,               1]
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])  
        self.homo_goal = homo_goal
        self.last_goal_update_time = time.time()
        goal_yaw = math.degrees(math.atan2(self.homo_goal[1, 0], self.homo_goal[0, 0]))
        print(f"[PLAN] new goal x={self.homo_goal[0, 3]:.3f}, y={self.homo_goal[1, 3]:.3f}, yaw={goal_yaw:.1f}deg")
        
    def move(self, vx, vy, vyaw):
        SPORT_API_ID_MOVE = 7105
        p = {"velocity": [float(vx), float(vy), float(vyaw)], "duration": 1}
        header = RequestHeader()
        identity = RequestIdentity()
        identity.api_id = SPORT_API_ID_MOVE
        header.identity = identity
        request = Request()
        request.header = header
        request.parameter = json.dumps(p)

        self.control_pub.publish(request)

        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.angular.z = float(vyaw)
        self.cmd_vel_pub.publish(twist)

    def _send_fsm_command(self, fsm_id, api_id=7101):
        p = {"data": fsm_id}
        header = RequestHeader()
        identity = RequestIdentity()
        identity.api_id = api_id
        header.identity = identity
        request = Request()
        request.header = header
        request.parameter = json.dumps(p)
        self.control_pub.publish(request)

    def initialize_g1(self):
        self.get_logger().info("G1 FSM: Entering Damp mode (ID=1)...")
        self._send_fsm_command(1)
        time.sleep(3.0)
        self.get_logger().info("G1 FSM: Standing up (ID=4)...")
        self._send_fsm_command(4)
        time.sleep(10.0)
        self.get_logger().info("G1 FSM: Starting locomotion (ID=500)...")
        self._send_fsm_command(500)
        time.sleep(3.0)
        self.get_logger().info("G1 initialization complete.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='G1 VLN client')
    parser.add_argument('--init-robot', type=str, default='false',
                        help='Initialize G1 by Damp -> StandUp -> Start. Use true only when needed.')
    args = parser.parse_args()
    init_robot = args.init_robot.lower() == 'true'

    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread)
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True

    rclpy.init()

    try:
        manager = G1VlnManager()
        if init_robot:
            print("[MAIN] Initializing G1: Damp -> StandUp -> Locomotion")
            manager.initialize_g1()
        else:
            print("[MAIN] Skipping robot initialization, assuming robot is already standing")

        control_thread_instance.start()
        planning_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        if manager is not None:
            manager.move(0.0, 0.0, 0.0)
            time.sleep(0.5)
            manager.destroy_node()
        rclpy.shutdown()

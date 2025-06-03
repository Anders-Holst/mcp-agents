import os
import re
from threading import Event

import yaml
from pyniryo2 import *
from pyniryo2 import PoseObject

DEFAULT_ROBOT_IP = '10.10.10.10'
HOME_POSE = PoseObject(x=0.1340, y=-0.0001, z=0.1649, roll=0.002, pitch=1.006, yaw=-0.001)
BASE_POSE_FILE = 'base-saved-poses.yaml'
LOCAL_POSE_FILE = 'local-saved-poses.yaml'

class Ned2:
    """
    Niryo Ned2 robot arm support object
    """
    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP):
        self._host = robot_ip
        self._setup_event = Event()
        self._move_event = Event()
        self._has_errors = False
        self._current_pose = HOME_POSE
        self._hold_torque = 30
        self.robot = None
        self.verbose = True
        self._pose_file = LOCAL_POSE_FILE
        self.base_poses = self._load_poses_from_yaml(BASE_POSE_FILE)
        self.poses = self._load_poses_from_yaml(self._pose_file)

    def open(self):
        self.robot = NiryoRobot(self._host)
        success = self._call_setup(self.robot.arm.calibrate_auto,
                                   self.__calibrate_success_callback,
                                   self.__calibrate_failure_callback)
        if success:
            success = self._call_setup(self.robot.tool.update_tool,
                                       self.__update_tool_success_callback,
                                       self.__update_tool_failure_callback)
        if not success:
            self.robot = None
        return success

    def close(self):
        robot = self.robot
        self.robot = None
        if robot:
            robot.arm.go_to_sleep()

    def is_open(self):
        return self.robot is not None

    def is_offline(self):
        return self.robot is None

    def _check_offline(self):
        if self.is_offline():
            print("Ned2: offline")
            return True
        return False

    def hardware_status(self):
        return self.robot.arm.hardware_status() if self.robot else 'Not connected'

    def joints_state(self):
        if self._check_offline():
            return None
        return self.robot.arm.joints_state() if self.robot else None

    def get_pose(self, name: str=None) -> PoseObject | None:
        if name is None:
            return self.robot.arm.get_pose() if self.robot else self._current_pose
        p = self.poses.get(name, None)
        if p:
            return p
        p = self.base_poses.get(name, None)
        if p:
            return p
        # Try to parse pose values
        if '[' in name or "PoseObject" in name:
            return self.pose_from_str(name)
        return None

    def set_pose(self, name: str) -> bool:
        """Save current pose locally with the specified name"""
        if name is None:
            return False
        self.poses[name] = self.get_pose()
        self._save_poses_to_yaml()
        return True

    def remove_pose(self, name: str) -> bool:
        """Remove the locally saved pose with the specified name"""
        if name not in self.poses:
            return False
        del self.poses[name]
        self._save_poses_to_yaml()
        return True

    def remove_all_poses(self) -> bool:
        """Remove all locally saved poses"""
        if not self.poses:
            return False
        self.poses = {}
        self._save_poses_to_yaml()
        return True

    def get_base_poses(self) -> dict[str, PoseObject]:
        return self.base_poses

    def get_local_poses(self) -> dict[str, PoseObject]:
        return self.poses

    def get_poses(self) -> dict[str, PoseObject]:
        return {**self.base_poses, **self.poses}

    def move_pose(self, pose, title=None) -> bool:
        return self._move_offline(pose, title) if self.is_offline() else self._move(self.robot.arm.move_pose, pose, title)

    def move_joints(self, joints, title=None) -> bool:
        if self._check_offline():
            return False
        return self._move(self.robot.arm.move_joints, joints, title)

    def move_to_home_pose(self):
        if self.is_offline():
            self._current_pose = HOME_POSE
        else:
            self.robot.arm.move_to_home_pose()

    def pick_from_pose(self, pose: PoseObject):
        if self._check_offline():
            return None
        return self.robot.pick_place.pick_from_pose(pose)

    def place_from_pose(self, pose: PoseObject):
        if self._check_offline():
            return None
        return self.robot.pick_place.place_from_pose(pose)

    def pick_and_place(self, source_pose: PoseObject, target_pose: PoseObject):
        if self._check_offline():
            return None
        return self.robot.pick_place.pick_and_place(source_pose, target_pose)

    def open_gripper(self):
        if self._check_offline():
            return None
        return self.robot.tool.open_gripper()

    def close_gripper(self):
        if self._check_offline():
            return None
        return self.robot.tool.close_gripper(hold_torque_percentage=self._hold_torque)

    def get_hold_torque(self):
        """Returns the current gripper hold torque in percent"""
        return self._hold_torque

    def set_hold_torque(self, percent):
        """Set the gripper hold torque in percent"""
        if self._check_offline():
            return False
        if 0 < percent <= 100:
            self._hold_torque = percent
            return True
        return False

    def get_max_speed(self):
        if self._check_offline():
            return 0
        return self.robot.arm.get_arm_max_velocity()

    def set_max_speed(self, percentage):
        if self._check_offline() and 0 < percentage <= 100:
            self.robot.arm.set_arm_max_velocity(percentage)
            return True
        return False

    def _call_setup(self, setup_function, success_callback, failure_callback) -> bool:
        self._setup_event.clear()
        setup_function(callback=success_callback, errback=failure_callback)
        self._setup_event.wait(30)
        return self._setup_event.is_set() and not self._has_errors

    def __calibrate_success_callback(self, result):
        if self.verbose:
            print('Ned2: Calibrate:', result['message'])
        self._setup_event.set()

    def __calibrate_failure_callback(self, result):
        self._has_errors = True
        print('Ned2: Error: Calibrate:', result['message'])
        self._setup_event.set()

    def __update_tool_success_callback(self, result):
        if self.verbose:
            print('Ned2: Update Tool:', result['message'])
        self._setup_event.set()

    def __update_tool_failure_callback(self, result):
        self._has_errors = True
        print('Ned2: Error: Update Tool:', result['message'])
        self._setup_event.set()

    def __move_callback(self, result):
        if result['status'] == 1:
            if self.verbose:
                print('Ned2:  move successful:', result['message'])
        else:
            self._has_errors = True
            print('Ned2:  Error: move failed:', result)
        self._move_event.set()

    def _move(self, move_function, target, title=None):
        self._has_errors = False
        self._move_event.clear()
        if title is not None:
            print('Ned2: Move to', title)
        move_function(target, callback=self.__move_callback)
        self._move_event.wait(20)
        if self._has_errors:
            return False
        if not self._move_event.is_set():
            print("*** timeout")
            return False
        self._current_pose = self.robot.arm.get_pose()
        if title is not None and self.robot:
            print('Ned2:  move done. Pose is', self.pose_to_str(self._current_pose))
        return True

    def _move_offline(self, pose, title=None):
        if title is not None:
            print('Ned2: Move to', title)
        self._current_pose = pose
        if title is not None:
            print('Ned2:  move done. Pose is', self.pose_to_str(pose))
        return True

    def _save_poses_to_yaml(self):
        with open(self._pose_file, 'w') as file:
            yaml.dump(self._convert_values_to_list(self.poses), file, default_flow_style=False)

    def _load_poses_from_yaml(self, filename):
        if not os.path.exists(filename):
            return {}
        try:
            with open(filename, 'r') as file:
                return self._convert_values_to_poses(yaml.safe_load(file)) or {}
        except (yaml.YAMLError, IOError) as e:
            print("Failed to read saved poses from {}: {}".format(filename, e))
            return {}

    def pose_from_str(self, input_text: str) -> PoseObject | None:
        try:
            float_pattern = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
            matches = re.findall(float_pattern, input_text)
            float_list = [float(x) for x in matches]
            if len(float_list) != 6:
                raise ValueError("Input string must contain exactly six float numbers.")
            return self._pose_from_list(float_list)
        except ValueError as e:
            print("Failed to parse pose {}: {}".format(input_text, e))
            return None

    @staticmethod
    def pose_to_str(pose: PoseObject):
        """Return a string representation of the pose object."""
        if not pose:
            return 'None'
        return ('PoseObject(x={:.4f}, y={:.4f}, z={:.4f}, roll={:.3f}, pitch={:.3f}, yaw={:.3f})'
                .format(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw))

    @staticmethod
    def _pose_from_list(p) -> PoseObject:
        """Convert a list of floats to a pose object."""
        return PoseObject(p[0], p[1], p[2], p[3], p[4], p[5])

    @staticmethod
    def _list_from_pose(pose: PoseObject):
        """Convert a pose object to a list of floats."""
        return [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]

    def _convert_values_to_list(self, poses):
        """Convert a dict with pose objects to a dict with lists of floats."""
        return {
            key: self._list_from_pose(value)
            for key, value in poses.items()
        }

    def _convert_values_to_poses(self, poses):
        """Convert a dict with list of floats to a dict with pose objects."""
        return {
            key: self._pose_from_list(value)
            for key, value in poses.items()
        }

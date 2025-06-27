import os

import yaml
from pyniryo import *

DEFAULT_ROBOT_IP = '10.10.10.10'
HOME_POSE = PoseObject(x=0.1340, y=-0.0001, z=0.1649, roll=0.002, pitch=1.006, yaw=-0.001, metadata=PoseMetadata.v1())
BASE_POSE_FILE = 'base-saved-poses.yaml'
LOCAL_POSE_FILE = 'local-saved-poses.yaml'

class Ned2:
    """
    Niryo Ned2 robot arm support object
    """
    robot: NiryoRobot | None

    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP):
        self._host = robot_ip
        self._current_pose = HOME_POSE
        self._hold_torque = 100
        self._manual_pick_and_place = False
        # Assume the default is 100 %
        self._arm_max_velocity = 100
        self.robot = None
        self.verbose = True
        self._pose_file = LOCAL_POSE_FILE
        self.base_poses = self._load_poses_from_yaml(BASE_POSE_FILE)
        self.poses = self._load_poses_from_yaml(self._pose_file)

    def open(self) -> bool:
        self.robot = NiryoRobot(self._host)
        self.robot.calibrate_auto()
        self.robot.update_tool()
        return True

    def close(self):
        robot = self.robot
        self.robot = None
        if robot:
            if robot.collision_detected:
                robot.clear_collision_detected()
            robot.go_to_sleep()
            robot.close_connection()

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
        return self.robot.get_hardware_status() if self.robot else 'Not connected'

    def get_joints(self):
        if self._check_offline():
            return None
        return self.robot.get_joints() if self.robot else None

    def get_pose(self, name: str=None) -> PoseObject | None:
        if name is None:
            return self.robot.get_pose() if self.robot else self._current_pose
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
        """Save the current pose locally with the specified name"""
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

    def move_pose(self, pose: PoseObject, title: str='') -> bool:
        if self.is_offline():
            return self._move_offline(pose, title)
        if title:
            print('Ned2: Move to', title)
        self.robot.move(pose)
        self._current_pose = self.robot.get_pose()
        if title:
            print('Ned2:  move done. Pose is', self.pose_to_str(self._current_pose))
        return True

    def move_joints(self, joints: JointsPosition, title: str='') -> bool:
        if self._check_offline():
            return False
        if title:
            print('Ned2: Move to', title)
        self.robot.move(joints)
        self._current_pose = self.robot.get_pose()
        if title:
            print('Ned2:  move done. Pose is', self.pose_to_str(self._current_pose))
        return True

    def move_to_home_pose(self):
        if self.is_offline():
            self._current_pose = HOME_POSE
        else:
            self.robot.move_to_home_pose()

    def pick_from_pose(self, pose: PoseObject):
        if self._check_offline():
            return None
        return self.robot.pick(pose)

    def place_from_pose(self, pose: PoseObject):
        if self._check_offline():
            return None
        return self.robot.place(pose)

    def pick_and_place(self, source_pose: PoseObject, target_pose: PoseObject, verbose: bool = False):
        if self._check_offline():
            return None
        if self._manual_pick_and_place:
            height_offset = 0.07
            pick_pose_high = source_pose.copy_with_offsets(z_offset=height_offset)
            pick_pose_high.metadata = source_pose.metadata
            place_pose_high = target_pose.copy_with_offsets(z_offset=height_offset)
            place_pose_high.metadata = target_pose.metadata
            if verbose:
                print("Move to", pick_pose_high)
            self.robot.move(pick_pose_high)
            self.open_gripper()
            if verbose:
              print("Move to", source_pose)
            self.robot.move(source_pose)
            self.close_gripper()
            if verbose:
                print("Move to", pick_pose_high)
            self.robot.move(pick_pose_high)
            if verbose:
                print("Move to", place_pose_high)
            self.robot.move(place_pose_high)
            if verbose:
                print("Move to", target_pose)
            self.robot.move(target_pose)
            self.open_gripper()
            if verbose:
                print("Move to", place_pose_high)
            return self.robot.move(place_pose_high)
        # Pick and place handled by the robot
        return self.robot.pick_and_place(source_pose, target_pose)

    def open_gripper(self):
        if self._check_offline():
            return None
        return self.robot.open_gripper(hold_torque_percentage=self._hold_torque)

    def close_gripper(self):
        if self._check_offline():
            return None
        return self.robot.close_gripper(hold_torque_percentage=self._hold_torque)

    def get_hold_torque(self):
        """Returns the current gripper hold torque in percent"""
        return self._hold_torque

    def set_hold_torque(self, percent):
        """Set the gripper hold torque in percent"""
        if 0 < percent <= 100:
            self._hold_torque = percent
            return True
        return False

    def get_manual_pick_and_place(self) -> bool:
        """Returns if pick-and-place is handled manually or by robot"""
        return False if self._check_offline() else self._manual_pick_and_place

    def set_manual_pick_and_place(self, manual: bool) -> bool:
        """Set if manual pick-and-place should be used or not"""
        if self._check_offline():
            return False
        self._manual_pick_and_place = manual
        return True

    def get_max_arm_velocity(self) -> int:
        """Returns the current arm maximum velocity in percent"""
        return 0 if self._check_offline() else self._arm_max_velocity

    def set_max_arm_velocity(self, percent):
        """Set the arm maximum velocity in percent"""
        if self._check_offline():
            return False
        if 0 < percent <= 100:
            self._arm_max_velocity = percent
            self.robot.set_arm_max_velocity(percent)
            return True
        return False

    @property
    def collision_detected(self):
        return self.robot.collision_detected if self.robot else False

    def clear_collision_detected(self):
        if self.is_open() and self.robot.collision_detected:
            self.robot.clear_collision_detected()
            self.move_to_home_pose()

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
        return PoseObject(p[0], p[1], p[2], p[3], p[4], p[5], metadata=PoseMetadata.v1())

    @staticmethod
    def _list_from_pose(pose: PoseObject):
        """Convert a pose object to a list of floats."""
        return [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]

    def _convert_values_to_list(self, poses):
        """Convert a dict with pose objects to a dict with lists of floats."""
        return {
            key: self._list_from_pose(value)
            for key, value in poses.items()
        } if poses else {}

    def _convert_values_to_poses(self, poses):
        """Convert a dict with a list of floats to a dict with pose objects."""
        return {
            key: self._pose_from_list(value)
            for key, value in poses.items()
        } if poses else {}

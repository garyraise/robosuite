import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class HarmonBracketGripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/harmon_bracket_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None

    @property
    def _important_sites(self):
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_site_cylinder",
            "ee": "ee",
            "ee_x": "ee_x",
            "ee_y": "ee_y",
            "ee_z": "ee_z",
            "tip": "tip",
        }

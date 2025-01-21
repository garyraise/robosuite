from collections import OrderedDict

import numpy as np

import os
os.environ["MUJOCO_GL"] = "glfw"
import dm_control.mujoco
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import PegsArena, EmptyArena, ChannelArena
from robosuite.models.objects import RoundNutObject, SquareNutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class AnchorChannel(ManipulationEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="HarmonBracketGripper",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        nut_type=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # task settings
        self.single_object_mode = single_object_mode

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        num_robots = len(robots)

        # Gripper
        gripper_types = self._input2list(gripper_types, num_robots)

        # Robot configurations to pass to super call
        robot_configs = [
            {
                "gripper_type": gripper_types[idx],
                "initial_qpos": np.zeros(6),
            }
            for idx in range(num_robots)
        ]

        # Run superclass init
        super(ManipulationEnv, self).__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="NullMount",
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            robot_configs=robot_configs,
            renderer=renderer,
            renderer_config=renderer_config,
        )

        self.deterministic_reset = True

        self.control_model = dm_control.mujoco.wrapper.core.MjModel(self.robots[0].robot_model.mujoco_model)
        self.physics = dm_control.mujoco.Physics.from_model(self.control_model)

    def reward(self, action=None):
        return 0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # load model for table top workspace
        mujoco_arena = ChannelArena()
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            #mujoco_objects=self.nuts,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()


    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _check_success(self):
        """
        Check if all nuts have been successfully placed around their corresponding pegs.

        Returns:
            bool: True if all nuts are placed correctly
        """
        return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest nut.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def reset(self):
        super().reset()

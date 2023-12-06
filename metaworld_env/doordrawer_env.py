import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for, full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerDoorDrawerEnv(SawyerXYZEnv):
    def __init__(self, task_id=0):
        self.task_id = task_id
        self.is_door = 1 - task_id // 2
        self.is_open = 1 - task_id % 2
        hand_low = (-1, 0., 0.05)
        hand_high = (1, 1, 0.5)
        obj_low = tuple([min(x, y) for x, y in zip((0., 0.85, 0.1), (-0.1, 0.9, 0.04))])
        obj_high = tuple([max(x, y) for x, y in zip((0.1, 0.95, 0.1), (0.1, 0.9, 0.04))])

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'door_init_angle': np.array([0.3, ]),
            'door_init_pos': np.array([0.1, 0.95, 0.1]),
            'drawer_init_angle': np.array([0.3, ], dtype=np.float32),
            'drawer_init_pos': np.array([0., 0.4, 0.04], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2]),
        }

        self.door_init_pos = self.init_config['door_init_pos']
        self.door_init_angle = self.init_config['door_init_angle']
        self.drawer_init_pos = self.init_config['drawer_init_pos']
        self.drawer_init_angle = self.init_config['drawer_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.door_angle_idx = self.model.get_joint_qpos_addr('doorjoint')
        self.drawer_angle_idx = self.model.get_joint_qpos_addr('goal_slidey')

    @property
    def model_name(self):
        return full_v2_path_for(f'sawyer_xyz/sawyer_door_drawer.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pullDist = self.compute_reward(action, ob)
        info = {
            'reachDist': reachDist,
            'goalDist': pullDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pullDist <= 0.08)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return np.concatenate(
            [self.data.get_geom_xpos('doorhandle').copy(), self.data.get_geom_xpos('drawerhandle').copy()])

    def _set_obj_xyz(self, pos1, pos2):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_angle_idx] = pos1
        qvel[self.door_angle_idx] = 0
        qpos[self.drawer_angle_idx] = pos2
        qvel[self.drawer_angle_idx] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def reset_model(self):
        self._reset_hand()

        self.door_init_pos = self._get_state_rand_vec()[:3] if self.random_init \
            else self.init_config['door_init_pos']
        self.drawer_init_pos = self._get_state_rand_vec()[3:] if self.random_init \
            else self.init_config['drawer_init_pos']
        obj_init_pos = self.drawer_init_pos if self.task_id // 2 else self.door_init_pos
        goal_delta = np.array([[-0.3, -0.25, 0.05], [0.1, -0.15, 0.05], [.0, -0.35, .0], [.0, -0.2, .0]])
        self._target_pos = obj_init_pos + goal_delta[self.task_id]

        self.sim.model.body_pos[self.model.body_name2id('door')] = self.door_init_pos
        self.sim.model.body_pos[self.model.body_name2id('drawer')] = self.drawer_init_pos
        self.sim.model.body_pos[self.model.body_name2id('drawer_cover')] = self.drawer_init_pos.copy()
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos
        if self.task_id == 3:
            self._set_obj_xyz(0, -0.2)  # drawer close
        elif self.task_id == 1:
            self._set_obj_xyz(-1.5708, 0)  # door close
        else:
            self._set_obj_xyz(0, 0)
        if self.task_id == 2:  # drawer open
            self.maxPullDist = 0.2
        elif self.task_id == 3:  # drawer close
            self.maxPullDist = np.abs(self.data.get_geom_xpos('drawerhandle')[1] - self._target_pos[1])
        else:
            self.maxPullDist = np.linalg.norm(self.data.get_geom_xpos('doorhandle')[:-1] - self._target_pos[:-1])
        self.target_reward = 1000 * self.maxPullDist + 1000 * 2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM = (rightFinger + leftFinger) / 2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions
        if self.is_door:
            objPos = obs[3:6]
        else:
            objPos = obs[6:9]
        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM = (rightFinger + leftFinger) / 2

        pullGoal = self._target_pos
        if self.is_door:
            pullDist = np.linalg.norm(objPos[:-1] - pullGoal[:-1])
        else:
            pullDist = np.abs(objPos[1] - pullGoal[1])
        reachDist = np.linalg.norm(objPos - fingerCOM)
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        def pullReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                        np.exp(-(pullDist ** 2) / c2) + np.exp(-(pullDist ** 2) / c3))
                pullRew = max(pullRew, 0)
                return pullRew
            else:
                return 0

        pullRew = pullReward()
        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]


class SawyerDoorDrawerEnv2(SawyerXYZEnv):
    def __init__(self, task_id=0):
        self.task_id = task_id
        self.is_door = 1 - task_id // 2
        self.is_open = 1 - task_id % 2
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = tuple([min(x, y) for x, y in zip((0., 0.85, 0.1), (-0.1, 0.9, 0.04))])
        obj_high = tuple([max(x, y) for x, y in zip((0.1, 0.95, 0.1), (0.1, 0.9, 0.04))])

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'door_init_angle': np.array([0.3, ]),
            'door_init_pos': np.array([0.1, 0.95, 0.1]),
            'drawer_init_angle': np.array([0.3, ], dtype=np.float32),
            'drawer_init_pos': np.array([0., 0.4, 0.04], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2]),
        }

        self.door_init_pos = self.init_config['door_init_pos']
        self.door_init_angle = self.init_config['door_init_angle']
        self.drawer_init_pos = self.init_config['drawer_init_pos']
        self.drawer_init_angle = self.init_config['drawer_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.door_angle_idx = self.model.get_joint_qpos_addr('doorjoint')
        self.drawer_angle_idx = self.model.get_joint_qpos_addr('goal_slidey')

    @property
    def model_name(self):
        return full_v2_path_for(f'sawyer_xyz/sawyer_door_drawer.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pullDist = self.compute_reward(action, ob)
        info = {
            'reachDist': reachDist,
            'goalDist': pullDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pullDist <= 0.08)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        door_pos = self.data.get_geom_xpos('doorhandle').copy()
        drawer_pos = self.get_body_com('drawer_link').copy() + np.array([.0, -.16, .0])
        if self.is_door:
            drawer_pos[0] -= 0.55
        else:
            door_pos[0] -= 0.55
        return np.concatenate([door_pos, drawer_pos])

    def _set_obj_xyz(self, pos1, pos2):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_angle_idx] = pos1
        qvel[self.door_angle_idx] = 0
        qpos[self.drawer_angle_idx] = pos2
        qvel[self.drawer_angle_idx] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def reset_model(self):
        self._reset_hand()

        self.door_init_pos = self._get_state_rand_vec()[:3] if self.random_init \
            else self.init_config['door_init_pos']
        self.drawer_init_pos = self._get_state_rand_vec()[3:] if self.random_init \
            else self.init_config['drawer_init_pos']
        obj_init_pos = self.drawer_init_pos if self.task_id // 2 else self.door_init_pos
        goal_delta = np.array([[-0.3, -0.45, 0.], [0.2, -0.2, 0.], [.0, -.36, .09], [.0, -.16, .09]])
        self._target_pos = obj_init_pos + goal_delta[self.task_id]

        self.sim.model.body_pos[self.model.body_name2id('door')] = self.door_init_pos
        self.sim.model.body_pos[self.model.body_name2id('drawer')] = self.drawer_init_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos
        if self.task_id == 3:
            self._set_obj_xyz(0, -0.15)  # drawer close
        elif self.task_id == 1:
            self._set_obj_xyz(-1.5708, 0)  # door close
        else:
            self._set_obj_xyz(0, 0)
        if self.task_id == 2:  # drawer open
            self.maxPullDist = 0.2
        elif self.task_id == 3:  # drawer close
            # self.maxPullDist = np.abs(self.data.get_geom_xpos('drawerhandle')[1] - self._target_pos[1])
            self.maxPullDist = 0.15
        else:
            self.maxPullDist = np.linalg.norm(self.data.get_geom_xpos('doorhandle')[:-1] - self._target_pos[:-1])
        self.target_reward = 1000 * self.maxPullDist + 1000 * 2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM = (rightFinger + leftFinger) / 2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions
        if self.is_door:
            objPos = obs[3:6]
        else:
            objPos = obs[6:9]
        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM = (rightFinger + leftFinger) / 2

        pullGoal = self._target_pos
        if self.is_door:
            pullDist = np.linalg.norm(objPos[:-1] - pullGoal[:-1])
        else:
            pullDist = np.abs(objPos[1] - pullGoal[1])
        reachDist = np.linalg.norm(objPos - fingerCOM)
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        def pullReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                        np.exp(-(pullDist ** 2) / c2) + np.exp(-(pullDist ** 2) / c3))
                pullRew = max(pullRew, 0)
                return pullRew
            else:
                return 0

        pullRew = pullReward()
        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]

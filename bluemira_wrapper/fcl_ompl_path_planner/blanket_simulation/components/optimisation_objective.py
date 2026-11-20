# SPDX-FileCopyrightText: 2026-present N.Salt
#
# SPDX-License-Identifier: MIT 2026
import numpy as np
import ompl.base as ob
from scipy.spatial.transform import Rotation


class BlanketTiltObjective(ob.OptimizationObjective):
    def __init__(self, si):
        super().__init__(si)
        self.space_information = si

    @staticmethod
    def _tilt(state):
        quaternion = state.rotation()
        _, pitch, roll = Rotation.from_quat([
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        ]).as_euler("ZYX")
        return abs(pitch) + abs(roll)

    def motionCost(self, s1, s2):
        state_space = self.space_information.getStateSpace()
        distance = state_space.distance(s1, s2)
        mean_tilt = (self._tilt(s1) + self._tilt(s2)) / 2
        return ob.Cost(mean_tilt * distance)


class MaximizeCenterlineTranslationObjective(ob.OptimizationObjective):
    def __init__(self, si, centerline_direction):
        super().__init__(si)
        direction = np.asarray(centerline_direction, dtype=float)
        self.centerline_direction = direction / np.linalg.norm(direction)

    def motionCost(self, s1, s2):
        start = np.array([s1.getX(), s1.getY(), s1.getZ()])
        end = np.array([s2.getX(), s2.getY(), s2.getZ()])
        progress = np.dot(end - start, self.centerline_direction)
        return ob.Cost(-progress)

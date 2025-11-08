"""In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
"""

from pid import PIDAgent
from keyframes import (
    hello,
    leftBackToStand,
    leftBellyToStand,
    rightBackToStand,
    rightBellyToStand,
    wipe_forehead,
)


class AngleInterpolationAgent(PIDAgent):
    def __init__(
        self,
        simspark_ip="localhost",
        simspark_port=3100,
        teamname="DAInamite",
        player_id=0,
        sync_mode=True,
    ):
        super(AngleInterpolationAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode
        )
        self.keyframes = ([], [], [])

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}

        # Extract keyframe data
        names, times, keys = keyframes
        if not names:
            return target_joints

        # Get current time
        current_time = perception.time

        # Iterate through each joint
        for i, joint_name in enumerate(names):
            joint_times = times[i]
            joint_keys = keys[i]

            # Skip if no keyframes for this joint
            if not joint_times:
                continue

            # Check if motion is complete
            if current_time > joint_times[-1]:
                # Motion finished, set to final position
                target_joints[joint_name] = joint_keys[-1][0]
                continue

            # Check if motion hasn't started
            if current_time < joint_times[0]:
                continue

            # Find the segment we're currently in
            for j in range(len(joint_times) - 1):
                t0 = joint_times[j]
                t1 = joint_times[j + 1]

                if t0 <= current_time <= t1:
                    # Get keyframe data
                    key0 = joint_keys[j]
                    key1 = joint_keys[j + 1]

                    # Extract angles and handles
                    angle0 = key0[0]
                    angle1 = key1[0]

                    # Extract Bezier handles if available
                    if len(key0) > 1 and len(key1) > 1:
                        # Handle2 of point 0 (outgoing handle)
                        handle0 = key0[2]  # [interpolation_type, dTime, dAngle]
                        # Handle1 of point 1 (incoming handle)
                        handle1 = key1[1]  # [interpolation_type, dTime, dAngle]

                        # Calculate control points for cubic Bezier
                        p0 = angle0
                        p3 = angle1
                        p1 = angle0 + handle0[2]  # angle + dAngle
                        p2 = angle1 + handle1[2]  # angle + dAngle

                        # Normalize time parameter t to [0, 1]
                        t = (current_time - t0) / (t1 - t0)

                        # Cubic Bezier interpolation: B(t) = (1-t)³p0 + 3(1-t)²t*p1 + 3(1-t)t²*p2 + t³*p3
                        interpolated_angle = (
                            (1 - t) ** 3 * p0
                            + 3 * (1 - t) ** 2 * t * p1
                            + 3 * (1 - t) * t**2 * p2
                            + t**3 * p3
                        )
                    else:
                        # Linear interpolation fallback
                        t = (current_time - t0) / (t1 - t0)
                        interpolated_angle = angle0 + t * (angle1 - angle0)

                    target_joints[joint_name] = interpolated_angle
                    break

        return target_joints


if __name__ == "__main__":
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()

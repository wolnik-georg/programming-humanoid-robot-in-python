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

from keyframes import (
    leftBackToStand,
    leftBellyToStand,
    wipe_forehead,
    rightBellyToStand,
    rightBackToStand,
)
from pid import PIDAgent
from keyframes import hello


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
        # Check if 'LHipYawPitch' exists in target_joints before copying to 'RHipYawPitch'
        if "LHipYawPitch" in target_joints:
            target_joints["RHipYawPitch"] = target_joints["LHipYawPitch"]
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        joint_targets = {}

        # Extract keyframe components
        joint_names, time_sequences, key_positions = keyframes

        # Early return for empty keyframes
        if not time_sequences or not key_positions:
            return joint_targets

        # Determine maximum duration across all joints
        try:
            max_duration = max(max(seq) for seq in time_sequences)
        except (TypeError, ValueError):
            max_duration = max(time_sequences) if time_sequences else 0

        # Compute cyclic time within keyframe duration
        current_time = perception.time % max_duration if max_duration > 0 else 0

        # Process each joint's motion
        for joint_idx, joint_name in enumerate(joint_names):
            time_points = time_sequences[joint_idx]
            position_keys = key_positions[joint_idx]

            # Handle single keyframe case
            if len(time_points) == 1:
                joint_targets[joint_name] = (
                    position_keys[0]
                    if isinstance(position_keys[0], (int, float))
                    else position_keys[0][0]
                )
                continue

            # Find appropriate time interval
            for interval_idx in range(len(time_points) - 1):
                start_time = time_points[interval_idx]
                end_time = time_points[interval_idx + 1]

                if start_time <= current_time <= end_time:
                    # Extract keyframe angles
                    start_angle = (
                        position_keys[interval_idx][0]
                        if isinstance(position_keys[interval_idx], list)
                        else position_keys[interval_idx]
                    )
                    end_angle = (
                        position_keys[interval_idx + 1][0]
                        if isinstance(position_keys[interval_idx + 1], list)
                        else position_keys[interval_idx + 1]
                    )

                    # Determine Bezier control points
                    if (
                        isinstance(position_keys[interval_idx], list)
                        and len(position_keys[interval_idx]) > 1
                        and isinstance(position_keys[interval_idx + 1], list)
                        and len(position_keys[interval_idx + 1]) > 2
                    ):

                        # Extract handle information
                        _, _, handle1_angle = position_keys[interval_idx][1]
                        _, _, handle2_angle = position_keys[interval_idx + 1][2]

                        # Calculate control points with enhanced curvature
                        control1 = start_angle + 1.5 * handle1_angle
                        control2 = end_angle + 1.5 * handle2_angle
                    else:
                        # Fallback control points for basic interpolation
                        angle_span = end_angle - start_angle
                        control1 = start_angle + 0.75 * angle_span
                        control2 = end_angle - 0.75 * angle_span

                    # Compute normalized parameter
                    interval_duration = end_time - start_time
                    time_in_interval = current_time - start_time
                    t_param = (
                        time_in_interval / interval_duration
                        if interval_duration > 0
                        else 0
                    )

                    # Cubic Bezier interpolation formula
                    angle_result = (
                        (1 - t_param) ** 3 * start_angle
                        + 3 * (1 - t_param) ** 2 * t_param * control1
                        + 3 * (1 - t_param) * t_param**2 * control2
                        + t_param**3 * end_angle
                    )

                    joint_targets[joint_name] = angle_result
                    break  # Found the correct interval

        return joint_targets


if __name__ == "__main__":
    agent = AngleInterpolationAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    # agent.keyframes = wipe_forehead()
    agent.run()

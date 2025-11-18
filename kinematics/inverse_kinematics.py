"""In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
"""

from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
import numpy as np


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        """solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        """
        angles = [0.0 for _ in range(6)]  # Initialize angles for 6 joints

        # Leg dimensions in meters
        hip_to_knee = 0.1  # Distance from hip to knee
        knee_to_ankle = 0.1029  # Distance from knee to ankle
        ankle_to_ground = 0.04519  # Foot height offset

        # Get desired foot position from transform
        foot_pos = transform[:3, 3]
        px, py, pz = foot_pos[0], foot_pos[1], foot_pos[2]

        # Adjust z-coordinate for foot height
        pz_corrected = pz + ankle_to_ground

        # Calculate distance from hip to corrected ankle position
        distance = np.sqrt(px**2 + py**2 + pz_corrected**2)

        # Determine KneePitch using cosine rule
        cos_theta_knee = (hip_to_knee**2 + knee_to_ankle**2 - distance**2) / (
            2 * hip_to_knee * knee_to_ankle
        )
        cos_theta_knee = np.clip(cos_theta_knee, -1, 1)  # Ensure valid cosine value
        theta_knee = np.arccos(cos_theta_knee)
        angles[3] = -theta_knee  # Negative for NAO joint convention

        # Determine HipPitch angle
        if distance >= hip_to_knee + knee_to_ankle:
            # Target unreachable, extend leg fully
            theta_hip = np.arctan2(-pz_corrected, np.sqrt(px**2 + py**2))
        elif distance <= abs(hip_to_knee - knee_to_ankle):
            # Target too close, bend leg fully
            theta_hip = np.arctan2(-pz_corrected, np.sqrt(px**2 + py**2)) + np.pi
        else:
            # Standard case
            alpha = np.arctan2(-pz_corrected, np.sqrt(px**2 + py**2))
            cos_beta = (hip_to_knee**2 + distance**2 - knee_to_ankle**2) / (
                2 * hip_to_knee * distance
            )
            cos_beta = np.clip(cos_beta, -1, 1)
            beta = np.arccos(cos_beta)
            theta_hip = alpha + beta

        angles[2] = theta_hip

        # Set AnklePitch to maintain foot parallel to ground
        angles[4] = -(theta_hip + theta_knee)

        # Approximate remaining angles
        angles[0] = 0.0  # HipYawPitch set to zero

        # HipRoll based on lateral position
        angles[1] = np.arctan2(py, px)

        # AnkleRoll to counter hip roll
        angles[5] = -angles[1]

        return angles

    def set_transforms(self, effector_name, transform):
        """solve the inverse kinematics and control joints use the results"""
        # Perform inverse kinematics calculation
        computed_angles = self.inverse_kinematics(effector_name, transform)

        # Select appropriate joint names for the effector
        if effector_name == "LLeg":
            joints = [
                "LHipYawPitch",
                "LHipRoll",
                "LHipPitch",
                "LKneePitch",
                "LAnklePitch",
                "LAnkleRoll",
            ]
        elif effector_name == "RLeg":
            joints = [
                "RHipYawPitch",
                "RHipRoll",
                "RHipPitch",
                "RKneePitch",
                "RAnklePitch",
                "RAnkleRoll",
            ]
        else:
            raise ValueError(f"Unknown effector name: {effector_name}")

        # Prepare keyframes with single time point
        time_points = [[0.0] for _ in joints]
        angle_lists = [[float(angle)] for angle in computed_angles]

        # Assign to keyframes tuple
        self.keyframes = (joints, time_points, angle_lists)


if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    print("Computed joint angles:", agent.keyframes[2])  # Print the angles
    # agent.run()  # Commented out to avoid connection

"""In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
"""

# add PYTHONPATH
import os
import sys

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "joint_control")
)

from numpy import sin, cos
from numpy.matlib import matrix, identity

from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(
        self,
        simspark_ip="localhost",
        simspark_port=3100,
        teamname="DAInamite",
        player_id=0,
        sync_mode=True,
    ):
        super(ForwardKinematicsAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode
        )
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {
            "Head": ["HeadYaw", "HeadPitch"],
            "LArm": ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll"],
            "RArm": ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"],
            "LLeg": [
                "LHipYawPitch",
                "LHipRoll",
                "LHipPitch",
                "LKneePitch",
                "LAnklePitch",
                "LAnkleRoll",
            ],
            "RLeg": [
                "RHipYawPitch",
                "RHipRoll",
                "RHipPitch",
                "RKneePitch",
                "RAnklePitch",
                "RAnkleRoll",
            ],
        }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        """calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        """
        T = identity(4)

        # Define rotation axes for each joint
        axis_map = {
            "HeadYaw": "z",
            "HeadPitch": "y",
            "LShoulderPitch": "y",
            "LShoulderRoll": "z",
            "LElbowYaw": "y",
            "LElbowRoll": "x",
            "RShoulderPitch": "y",
            "RShoulderRoll": "z",
            "RElbowYaw": "y",
            "RElbowRoll": "x",
            "LHipYawPitch": "z",
            "LHipRoll": "x",
            "LHipPitch": "y",
            "LKneePitch": "y",
            "LAnklePitch": "y",
            "LAnkleRoll": "x",
            "RHipYawPitch": "z",
            "RHipRoll": "x",
            "RHipPitch": "y",
            "RKneePitch": "y",
            "RAnklePitch": "y",
            "RAnkleRoll": "x",
        }

        # Define translations (link lengths) for each joint
        trans_map = {
            "HeadYaw": [0.0, 0.0, 0.1265],
            "HeadPitch": [0.0, 0.0, 0.0],
            "LShoulderPitch": [0.0, 0.098, 0.1],
            "LShoulderRoll": [0.0, 0.0, 0.0],
            "LElbowYaw": [0.105, 0.015, 0.0],
            "LElbowRoll": [0.05595, 0.0, 0.0],
            "RShoulderPitch": [0.0, -0.098, 0.1],
            "RShoulderRoll": [0.0, 0.0, 0.0],
            "RElbowYaw": [0.105, -0.015, 0.0],
            "RElbowRoll": [0.05595, 0.0, 0.0],
            "LHipYawPitch": [0.0, 0.05, -0.085],
            "LHipRoll": [0.0, 0.0, 0.0],
            "LHipPitch": [0.0, 0.0, 0.0],
            "LKneePitch": [0.0, 0.0, -0.1],
            "LAnklePitch": [0.0, 0.0, 0.0],
            "LAnkleRoll": [0.0, 0.0, -0.1029],
            "RHipYawPitch": [0.0, -0.05, -0.085],
            "RHipRoll": [0.0, 0.0, 0.0],
            "RHipPitch": [0.0, 0.0, 0.0],
            "RKneePitch": [0.0, 0.0, -0.1],
            "RAnklePitch": [0.0, 0.0, 0.0],
            "RAnkleRoll": [0.0, 0.0, -0.1029],
        }

        axis = axis_map.get(joint_name, "z")  # default to z if not found
        tx, ty, tz = trans_map.get(joint_name, [0.0, 0.0, 0.0])

        # Apply rotation based on axis
        if axis == "x":
            T[1, 1] = cos(joint_angle)
            T[1, 2] = -sin(joint_angle)
            T[2, 1] = sin(joint_angle)
            T[2, 2] = cos(joint_angle)
        elif axis == "y":
            T[0, 0] = cos(joint_angle)
            T[0, 2] = sin(joint_angle)
            T[2, 0] = -sin(joint_angle)
            T[2, 2] = cos(joint_angle)
        elif axis == "z":
            T[0, 0] = cos(joint_angle)
            T[0, 1] = -sin(joint_angle)
            T[1, 0] = sin(joint_angle)
            T[1, 1] = cos(joint_angle)

        # Apply translation
        T[0, 3] = tx
        T[1, 3] = ty
        T[2, 3] = tz

        return T

    def forward_kinematics(self, joints):
        """forward kinematics

        :param joints: {joint_name: joint_angle}
        """
        for chain_joints in self.chains.values():
            T = identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                T = T @ Tl
                self.transforms[joint] = T


if __name__ == "__main__":
    agent = ForwardKinematicsAgent()
    agent.run()

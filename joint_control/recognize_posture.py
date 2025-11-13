"""In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

"""

from angle_interpolation import AngleInterpolationAgent
from keyframes import leftBellyToStand, rightBackToStand, leftBackToStand
from keyframes import hello
import pickle
import numpy as np
import os


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(
        self,
        simspark_ip="localhost",
        simspark_port=3100,
        teamname="DAInamite",
        player_id=0,
        sync_mode=True,
    ):
        super(PostureRecognitionAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode
        )
        self.posture = "unknown"
        self.posture_classifier = self.load_classifier("robot_pose.pkl")

    def load_classifier(self, filename):
        current_dir = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(current_dir, filename)

        with open(file_path, "rb") as file:
            classifier = pickle.load(file)
        return classifier

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = "unknown"
        joint_angles = [
            perception.joint["LHipYawPitch"],
            perception.joint["LHipPitch"],
            perception.joint["LHipRoll"],
            perception.joint["LKneePitch"],
            perception.joint["LShoulderPitch"],
            perception.joint["RHipYawPitch"],
            perception.joint["RHipPitch"],
            perception.joint["RHipRoll"],
            perception.imu[0],
            perception.imu[1],
        ]

        input_data = np.array(joint_angles).reshape(1, -1)
        predicted_class = self.posture_classifier.predict(input_data)[0]
        classes = [
            "HeadBack",
            "Stand",
            "Left",
            "Sit",
            "Back",
            "StandInit",
            "Right",
            "Crouch",
            "Belly",
            "Frog",
            "Knee",
        ]

        posture = classes[predicted_class]

        return posture


if __name__ == "__main__":
    agent = PostureRecognitionAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()

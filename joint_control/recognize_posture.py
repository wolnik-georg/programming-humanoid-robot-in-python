"""In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

"""

from angle_interpolation import AngleInterpolationAgent
from keyframes import (
    hello,
    leftBackToStand,
    leftBellyToStand,
    rightBackToStand,
    rightBellyToStand,
    wipe_forehead,
)
import pickle
from os import listdir

ROBOT_POSE_DATA_DIR = "robot_pose_data"
ROBOT_POSE_CLF = "robot_pose.pkl"


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
        self.posture_classifier = pickle.load(open(ROBOT_POSE_CLF, "rb"))
        self.classes = listdir(ROBOT_POSE_DATA_DIR)

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        print(f"Recognized posture: {self.posture}")  # Add this line to see output
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = "unknown"
        # Extract features in the same order as training data
        features = [
            perception.joint["LHipYawPitch"],
            perception.joint["LHipRoll"],
            perception.joint["LHipPitch"],
            perception.joint["LKneePitch"],
            perception.joint["RHipYawPitch"],
            perception.joint["RHipRoll"],
            perception.joint["RHipPitch"],
            perception.joint["RKneePitch"],
            perception.imu[0],  # AngleX
            perception.imu[1],  # AngleY
        ]
        # Predict the class index
        predicted_class = self.posture_classifier.predict([features])[0]
        # Map to posture name
        posture = self.classes[predicted_class]
        return posture


if __name__ == "__main__":
    agent = PostureRecognitionAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()

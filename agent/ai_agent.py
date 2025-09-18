# agent/ai_agent.py

import airsim
import time
import random

class RandomAgent:
    def __init__(self, client):
        self.client = client
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def explore(self, duration=60):
        start_time = time.time()
        while time.time() - start_time < duration:
            vx = random.uniform(-3, 3)
            vy = random.uniform(-3, 3)
            vz = random.uniform(-1, 1)
            self.client.moveByVelocityAsync(vx, vy, vz, 1).join()
            time.sleep(0.5)

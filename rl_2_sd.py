"""
https://www.youtube.com/watch?v=RznKVRTFkBY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL
"""
import socket
import json
import numpy as np
import random


def sendAction(s, action, json_data):
    if action == 0:
        s.sendall(b'FORWARD')
    elif action == 1:
        s.sendall(b'RIGHT')
    elif action == 2:
        s.sendall(b'LEFT')

    sensor_distance, traveled_distance, status, partial_distance = getObservation(json_data)
    if (status == -1):
        done = True
    else:
        done = False

    return sensor_distance, traveled_distance, done, partial_distance


def getObservation(json_data):
    print(json_data)
    data = json.loads(json_data)
    return np.array(data["Distance"]).reshape(1, ), data["TraveledDistance"], data["Status"], data["PartialDistance"]


HOST = 'localhost'  # The remote host
PORT = 4449  # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
s.sendall(b'Starting connection message')

total_steps = 0

for i_episode in range(500):

    sensor_distance, traveled_distance, status, partial_distance = getObservation(s.recv(1024))

    ep_r = 0

    while True:
        action = random.randint(0, 3)

        # observation_, reward, done, info = env.step(action)
        observation_, traveled_distance, done, partial_distance = sendAction(s, action, s.recv(1024))

        reward = partial_distance

        # RL.store_transition(sensor_distance, action, reward, observation_)

        ep_r += reward
        # if total_steps > 1000:  # learn after 1000 steps
        # RL.learn()
        # print("LEARNT")

        if done:
            steps_ = 0
            print('episode: ', i_episode,
                  ' ep_r: ', round(ep_r, 2),
                  #' epsilon: ', round(RL.epsilon, 2),
                  ' total_steps: ', total_steps)
            break

        observation = observation_
        total_steps += 1

s.close()

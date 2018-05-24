from RL_Network import DeepQNetwork
import socket
import json
import numpy as np

num_epochs = 1000


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
    json_data = json_data.decode("utf-8")
    k1 = json_data.rfind("{")
    k2 = json_data.rfind("}")
    new_string = json_data[k1:(k2 + 1)]

    data = json.loads(new_string)

    return np.array(data["Distance"]).reshape(1, ), data["TraveledDistance"], data["Status"], data["PartialDistance"]


def run():
    step = 0
    for episode in range(num_epochs):
        print("--- EPISODE: ", episode)
        # initial observation
        # observation = env.reset()
        zero_counters = 0
        sensor_distance, traveled_distance, status, partial_distance = getObservation(s.recv(1024))

        observation = sensor_distance
        prev = 0
        while True:
            # fresh env
            # env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            # observation_, reward, done = env.step(action)
            observation_, traveled_distance, done, partial_distance = sendAction(s, action, s.recv(1024))

            # calculate reward with a single sensor in front
            if done:
                reward = -200
            else:
                reward = partial_distance

            if reward == 0 and prev == 0:
                zero_counters += 1
            elif prev == 0:
                zero_counters = 0

            if zero_counters <= 4 and not done:
                reward *= observation_

            print(reward, " _ sensor_distance: ", observation_, "(", zero_counters, ")")

            prev = partial_distance

            RL.store_transition(observation, action, reward, observation_)

            """
            if (step > 100) and (step % 5 == 0):
                RL.learn()
            """

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        """
        if episode % 20 == 0:
            RL.store_params_temp()
        """
    # end of game
    print('Game over')


if __name__ == "__main__":
    HOST = 'localhost'  # The remote host
    PORT = 4449  # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(b'Starting connection message')

    nb_actions = 3
    nb_features = 1

    RL = DeepQNetwork(nb_actions, nb_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      # e_greedy=0.9, #90% chooses the best action
                      replace_target_iter=100,
                      memory_size=2000,
                      restore=True
                      # output_graph=True
                      )
    run()
    # RL.store_params()
    RL.plot_cost()

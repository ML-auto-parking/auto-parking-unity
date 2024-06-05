# 이미지 데이터 제거
import numpy as np
import random
import copy
import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import matplotlib.pyplot as plt

for _ in range(10):
    # DQN을 위한 파라미터 값 세팅
    action_size = 3  # agent가 취할 수 있는 행동의 개수

    WHEEL_TORQUE_MAX = 21
    STEERING_ANGLE_MAX = 21

    action_dim = WHEEL_TORQUE_MAX + STEERING_ANGLE_MAX  # agent가 취할 수 있는 행동의 차원

    load_model = False
    train_mode = True

    VECTOR_OBS = 0
    IMAGE_OBS = 1
    AGENT_POS_OBS = 2

    batch_size = 64
    mem_maxlen = 1000
    discount_factor = 0.9
    learning_rate = 0.00025

    train_start_step = 5000  # 초기 탐험
    first_step = 80000 + train_start_step
    run_step = first_step if train_mode else 0  # 훈련 스텝
    test_step = 1  # 테스트 스텝
    target_update_step = 100

    print_interval = 10
    save_interval = 50

    epsilon_eval = 0.05
    epsilon_init = 1.0 if train_mode else epsilon_eval
    epsilon_min = 0.1
    explore_step = run_step * 0.8
    epsilon_delta = (epsilon_init - epsilon_min) / explore_step if train_mode else 0.

    # 유니티 환경 경로
    game = "AutoParking"
    version = 131
    env_name = f'../Env/ap-{version}'

    # 모델 저장 및 불러오기 경로
    date_time = datetime.datetime.now().strftime("%m-%d %H:%M")
    save_path = f"./saved_models/{game}/DQN/a2-s2/ver-{version}/{date_time}"
    load_path = f"./saved_models/{game}/DQN/a2-s2/ver-{version}/06-05 04:07"
    # 20240602012700

    # 연산 장치
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    class DQN(torch.nn.Module):
        def __init__(self, vector_dim, agent_pos_dim, action_dim):
            super(DQN, self).__init__()
            linear_input_size = vector_dim + agent_pos_dim

            self.fc1 = torch.nn.Linear(linear_input_size, 512)
            self.fc2 = torch.nn.Linear(512, 512)
            self.q = torch.nn.Linear(512, action_dim)

        def forward(self, vector, agent_pos):
            x = torch.cat((vector, agent_pos), dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.q(x)


    class PrioritizedReplayBuffer:
        def __init__(self, capacity, alpha=0.6):
            self.capacity = capacity
            self.alpha = alpha
            self.memory = []
            self.pos = 0
            self.priorities = np.zeros((capacity,), dtype=np.float32)

        def append(self, transition):
            max_prio = self.priorities.max() if self.memory else 1.0

            if len(self.memory) < self.capacity:
                self.memory.append(transition)
            else:
                self.memory[self.pos] = transition

            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

        def sample(self, batch_size, beta=0.4):
            if len(self.memory) == self.capacity:
                prios = self.priorities
            else:
                prios = self.priorities[:self.pos]

            probs = prios ** self.alpha
            probs /= probs.sum()

            # NaN 값 검사 및 조정
            if np.isnan(probs).any():
                probs = np.nan_to_num(probs, nan=1e-6)
                probs /= probs.sum()

            indices = np.random.choice(len(self.memory), batch_size, p=probs)
            samples = [self.memory[idx] for idx in indices]

            total = len(self.memory)
            weights = (total * probs[indices]) ** (-beta)
            weights /= weights.max()
            return samples, indices, np.array(weights, dtype=np.float32)

        def update_priorities(self, batch_indices, batch_priorities):
            for idx, prio in zip(batch_indices, batch_priorities):
                self.priorities[idx] = prio


    # DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의
    class DQNAgent:
        def __init__(self, vector_dim, agent_pos_dim):
            self.network = DQN(vector_dim, agent_pos_dim, action_dim).to(device)
            self.target_network = copy.deepcopy(self.network)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
            self.memory = PrioritizedReplayBuffer(mem_maxlen)
            self.epsilon = epsilon_init
            self.writer = SummaryWriter(save_path)

            if load_model:
                print(f"... Load Model from {load_path}/ckpt ...")
                checkpoint = torch.load(load_path + '/ckpt', map_location=device)
                self.network.load_state_dict(checkpoint["network"])
                self.target_network.load_state_dict(checkpoint["network"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

        def get_action(self, vector, agent_pos, training=True):
            self.network.train(training)
            epsilon = self.epsilon if training else epsilon_eval

            if epsilon > random.random():
                wheel_torque = np.random.randint(0, WHEEL_TORQUE_MAX, size=(vector.shape[0], 1))  # 휠 토크 (11개의 이산 값)
                steering_angle = np.random.randint(0, STEERING_ANGLE_MAX,
                                                   size=(vector.shape[0], 1))  # 조향 각도 (11개의 이산 값)
                brake_state = np.random.randint(0, 1, size=(vector.shape[0], 1))  # 브레이크 상태 (2개의 이산 값)
                action = np.concatenate([wheel_torque, steering_angle, brake_state], axis=1)

                # print(f"Random Action: {action}")
                # print(f"Wheel Torque: {wheel_torque}, Steering Angle: {steering_angle}, Brake State: {brake_state}")

            else:
                with torch.no_grad():
                    vector_tensor = torch.FloatTensor(vector).to(device)
                    agent_pos_tensor = torch.FloatTensor(agent_pos).to(device)
                    q = self.network(vector_tensor, agent_pos_tensor).squeeze(1)

                    # print(f"Q-values: {q.cpu().numpy()}")

                    # 액션 계산
                    wheel_torque_q_values = q[:, :WHEEL_TORQUE_MAX]
                    steering_angle_q_values = q[:, WHEEL_TORQUE_MAX:WHEEL_TORQUE_MAX + STEERING_ANGLE_MAX]

                    wheel_torque = torch.argmax(wheel_torque_q_values, dim=1).cpu().numpy()
                    steering_angle = torch.argmax(steering_angle_q_values, dim=1).cpu().numpy()

                    # print(f"Clamped Wheel Torque: {wheel_torque}, Q-values: {wheel_torque_q_values.cpu().numpy()}")
                    # print(f"Clamped Steering Angle: {steering_angle}, Q-values: {steering_angle_q_values.cpu().numpy()}")
                    # print(f"Clamped Brake State: {brake_state}, Q-values: {brake_state_q_values.cpu().numpy()}")

                    action = np.stack([wheel_torque, steering_angle, [0]], axis=1)
            print(f"Action: {action}")

            return action

        def append_sample(self, vector, agent_pos, action, reward, next_vector, next_agent_pos, done):
            self.memory.append((vector, agent_pos, action, reward, next_vector, next_agent_pos, done))

        def train_model(self, beta=0.4):
            samples, indices, weights = self.memory.sample(batch_size, beta)

            # 데이터를 각각의 배열로 분할
            vectors = np.array([sample[0] for sample in samples])
            agent_pos = np.array([sample[1] for sample in samples])
            actions = np.array([sample[2] for sample in samples])
            rewards = np.array([sample[3] for sample in samples])
            next_vectors = np.array([sample[4] for sample in samples])
            next_agent_pos = np.array([sample[5] for sample in samples])
            dones = np.array([sample[6] for sample in samples])

            vectors, agent_pos, actions, rewards, next_vectors, next_agent_pos, dones = map(
                lambda x: torch.FloatTensor(x).to(device),
                [vectors, agent_pos, actions, rewards, next_vectors, next_agent_pos, dones])

            if vectors.ndim == 3:
                vectors = vectors.squeeze(1)
            if next_vectors.ndim == 3:
                next_vectors = next_vectors.squeeze(1)

            if agent_pos.ndim == 3:
                agent_pos = agent_pos.squeeze(1)
            if next_agent_pos.ndim == 3:
                next_agent_pos = next_agent_pos.squeeze(1)

            vectors = torch.FloatTensor(vectors).to(device)
            next_vectors = torch.FloatTensor(next_vectors).to(device)
            agent_pos = torch.FloatTensor(agent_pos).to(device)
            next_agent_pos = torch.FloatTensor(next_agent_pos).to(device)
            actions = torch.FloatTensor(actions).to(device)  # actions를 FloatTensor로 변환
            rewards = torch.FloatTensor(rewards).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # Q(s, a) 계산
            actions = actions.squeeze(1).long()

            # 각 행동에 대해 one-hot 인코딩 수행
            wheel_torque_one_hot = F.one_hot(actions[:, 0], num_classes=WHEEL_TORQUE_MAX).to(device)
            steering_angle_one_hot = F.one_hot(actions[:, 1], num_classes=STEERING_ANGLE_MAX).to(device)

            # print(f"wheel_torque_one_hot: {wheel_torque_one_hot}")
            # print(f"steering_angle_one_hot: {steering_angle_one_hot}")
            # print(f"brake_state_one_hot: {brake_state_one_hot}")

            # one-hot 인코딩된 행동을 결합
            one_hot_action = torch.cat((wheel_torque_one_hot, steering_angle_one_hot), dim=1).float()
            q = (self.network(vectors, agent_pos) * one_hot_action).sum(dim=1, keepdim=True)

            # print(f"one_hot_action: {one_hot_action}")

            # Q(s', a') 계산
            with torch.no_grad():
                target_q_values = self.target_network(next_vectors, next_agent_pos).detach()
                max_q_values = torch.max(target_q_values, dim=1, keepdim=True)[0]
                target_q = rewards + (1 - dones) * discount_factor * max_q_values

            # print(f"target_q_values: {target_q_values}")
            # print(f"target_q_values shape: {target_q_values.shape}")
            # print("done: ", dones)
            # print(f"dones shape: {dones.shape}")
            # print(f"max_q_values: {max_q_values}")
            # print(f"max_q_values shape: {max_q_values.shape}")
            # print(f"rewards: {rewards}")
            # print(f"rewards shape: {rewards.shape}")
            # print(f"target_q: {target_q}")
            # print(f"target_q shape: {target_q.shape}")
            # print(f"q values: {q}")
            # print(f"q values shape: {q.shape}")

            td_errors = target_q - q
            loss = F.smooth_l1_loss(q, target_q)

            # print(f"TD Errors: {td_errors}")
            # print(f"Loss: {loss}")

            # 모델 최적화
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # TD 오차로 우선순위 업데이트
            td_errors_np = td_errors.data.cpu().numpy().squeeze()
            td_errors_np = np.abs(td_errors_np) + 1e-6  # NaN 방지 및 소수 값 보정
            self.memory.update_priorities(indices, td_errors_np.flatten())

            # 엡실론 감소
            self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)

            return loss.item()

        def update_target(self):
            self.target_network.load_state_dict(self.network.state_dict())

        def save_model(self):
            print(f"... Save Model to {save_path}/ckpt ...")
            torch.save({
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, save_path + '/ckpt')

        def write_summary(self, score, loss, epsilon, step):
            self.writer.add_scalar("run/score", score, step)
            self.writer.add_scalar("model/loss", loss, step)
            self.writer.add_scalar("model/epsilon", epsilon, step)


    # Main 함수
    if __name__ == '__main__':
        engine_configuration_channel = EngineConfigurationChannel()
        env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel], worker_id=7)
        env.reset()

        behavior_name = list(env.behavior_specs.keys())[0]
        spec = env.behavior_specs[behavior_name]
        engine_configuration_channel.set_configuration_parameters(time_scale=20.0, target_frame_rate=60)
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        vector_obs = decision_steps.obs[VECTOR_OBS]
        agent_pos_obs = decision_steps.obs[AGENT_POS_OBS]

        vector_size = vector_obs.shape[1]
        agents_pos_dim = agent_pos_obs.shape[1]

        agent = DQNAgent(vector_size, agents_pos_dim)

        losses, scores, episode, score = [], [], 0, 0
        for step in range(run_step + test_step):
            if step == run_step:
                if train_mode:
                    agent.save_model()
                print("TEST START")
                train_mode = False
                engine_configuration_channel.set_configuration_parameters(time_scale=1.0, target_frame_rate=60)

            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if len(decision_steps) > 0:
                vector = decision_steps.obs[VECTOR_OBS]  # 벡터 데이터
                agent_pos = decision_steps.obs[AGENT_POS_OBS]  # 에이전트 위치
            else:
                vector = terminal_steps.obs[VECTOR_OBS]
                agent_pos = terminal_steps.obs[AGENT_POS_OBS]

            if len(decision_steps) > 0:  # 에이전트가 존재하는지 확인
                action = agent.get_action(vector, agent_pos, train_mode)
                action_tuple = ActionTuple()
                action_tuple.add_discrete(action)  # 이산적 행동 (휠 토크, 조향 각도, 브레이크 상태)

                env.set_actions(behavior_name, action_tuple)
                env.step()

                decision_steps, terminal_steps = env.get_steps(behavior_name)  # 각 스텝에서의 보상을 가져옴
            else:
                env.step()

            # print("agent_pos: ", agent_pos)
            # print("agent_pos shape: ", agent_pos.shape)
            # print("vector: ", vector)
            # print("vector shape: ", vector.shape)

            done = len(terminal_steps) > 0
            reward = terminal_steps.reward if done else decision_steps.reward
            next_vector = terminal_steps.obs[VECTOR_OBS] if done else decision_steps.obs[VECTOR_OBS]
            next_agent_pos = terminal_steps.obs[AGENT_POS_OBS] if done else decision_steps.obs[AGENT_POS_OBS]

            # print("terminal_steps reward: ", terminal_steps.reward)
            # print("decision_steps reward: ", decision_steps.reward)
            score += reward[0]

            # Debug print statements
            print(f'Step: {step}, Reward: {reward[0]}, Score: {score}, Epsilon: {agent.epsilon:.4f}')

            if train_mode:
                agent.append_sample(vector, agent_pos, action, reward, next_vector, next_agent_pos, [done])

            if train_mode and step > max(batch_size, train_start_step):
                loss = agent.train_model()
                losses.append(loss)

                if step % target_update_step == 0:
                    agent.update_target()

            if done:
                episode += 1
                scores.append(score)
                score = 0

                # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록
                if episode % print_interval == 0:
                    mean_score = np.mean(scores)
                    mean_loss = np.mean(losses)
                    agent.write_summary(mean_score, mean_loss, agent.epsilon, step)
                    losses, scores = [], []

                    print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " + \
                          f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

                # 네트워크 모델 저장
                if train_mode and episode % save_interval == 0:
                    agent.save_model()

        env.close()

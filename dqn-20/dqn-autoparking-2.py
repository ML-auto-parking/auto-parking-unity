import numpy as np
import random
import copy
import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# DQN을 위한 파라미터 값 세팅
action_size = 3  # 연속적인 행동 두 개 (휠 토크, 조향 각도) + 이산적인 행동 하나 (브레이크 상태)
action_dim = 24

load_model = False
train_mode = True

batch_size = 64
mem_maxlen = 10000
discount_factor = 0.9
learning_rate = 0.00025

run_step = 50000 if train_mode else 0 # 훈련 스텝
test_step = 10000 # 테스트 스텝
train_start_step = 10000 # 초기 탐험
target_update_step = 100

print_interval = 10
save_interval = 100

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.8
epsilon_delta = (epsilon_init - epsilon_min) / explore_step if train_mode else 0.

# 유니티 환경 경로
game = "AutoParking"
version = 21
env_name = f'../Env/ap-{version}'

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/DQN/{date_time}"
load_path = f"./saved_models/{game}/DQN/-"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(torch.nn.Module):
    def __init__(self, image_dim, vector_dim, action_size, action_dim):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=image_dim[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        print('image_dim: ', image_dim)
        print('vector_dim: ', vector_dim)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(image_dim[1], 8, 4)
        convh = conv2d_size_out(image_dim[2], 8, 4)
        convw = conv2d_size_out(convw, 4, 2)
        convh = conv2d_size_out(convh, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)
        convh = conv2d_size_out(convh, 3, 1)

        conv_output_size = convw * convh * 128
        linear_input_size = conv_output_size + vector_dim
        print(f"convw: {convw}, convh: {convh}, conv_output_size: {conv_output_size}, linear_input_size: {linear_input_size}")

        self.fc1 = torch.nn.Linear(linear_input_size, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.q = torch.nn.Linear(512, action_dim)
        print('action_dim: ', action_dim)

    def forward(self, image, vector):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, vector), dim=1)
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
    def __init__(self, image_dim, vector_dim):
        self.network = DQN(image_dim, vector_dim, action_size, action_dim).to(device)
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

    def get_action(self, image, vector, training=True):
        self.network.eval()
        epsilon = self.epsilon if training else epsilon_eval

        if epsilon > random.random():
            wheel_torque = np.random.randint(0, 11, size=(vector.shape[0], 1))  # 휠 토크 (11개의 이산 값)
            steering_angle = np.random.randint(0, 11, size=(vector.shape[0], 1))  # 조향 각도 (11개의 이산 값)
            brake_state = np.random.randint(0, 2, size=(vector.shape[0], 1))  # 브레이크 상태 (2개의 이산 값)
            action = np.concatenate([wheel_torque, steering_angle, brake_state], axis=1)
        else:
            with torch.no_grad():
                image_tensor = torch.FloatTensor(image).to(device).permute(0, 3, 1,
                                                                           2)  # (batch, height, width, channels) -> (batch, channels, height, width)
                vector_tensor = torch.FloatTensor(vector).to(device)
                q = self.network(image_tensor, vector_tensor)

                # q-values에서 이산 행동을 추출
                wheel_torque = torch.clamp((q[:, 0] * 4).round().int(), 0, 4).cpu().numpy()  # 휠 토크 (0~4)
                steering_angle = torch.clamp((q[:, 1] * 4).round().int(), 0, 4).cpu().numpy()  # 조향 각도 (0~4)
                brake_state = torch.clamp(q[:, 2].round().int(), 0, 1).cpu().numpy()  # 브레이크 상태 (0 또는 1)

                action = np.stack([wheel_torque, steering_angle, brake_state], axis=1)

        return action

    def append_sample(self, image, vector, action, reward, next_image, next_vector, done):
        self.memory.append((image, vector, action, reward, next_image, next_vector, done))

    def train_model(self, beta=0.4):
        samples, indices, weights = self.memory.sample(batch_size, beta)

        # 데이터를 각각의 배열로 분할
        images = np.array([sample[0] for sample in samples])
        vectors = np.array([sample[1] for sample in samples])
        actions = np.array([sample[2] for sample in samples])
        rewards = np.array([sample[3] for sample in samples])
        next_images = np.array([sample[4] for sample in samples])
        next_vectors = np.array([sample[5] for sample in samples])
        dones = np.array([sample[6] for sample in samples])

        # Tensor로 변환
        if images.ndim == 5:
            images = images.squeeze(1)  # (batch, 1, height, width, channels) -> (batch, height, width, channels)
        if next_images.ndim == 5:
            next_images = next_images.squeeze(1)

        if vectors.ndim == 3:
            vectors = vectors.squeeze(1)
        if next_vectors.ndim == 3:
            next_vectors = next_vectors.squeeze(1)

        images = torch.FloatTensor(images).to(device).permute(0, 3, 1, 2)
        next_images = torch.FloatTensor(next_images).to(device).permute(0, 3, 1, 2)
        vectors = torch.FloatTensor(vectors).to(device).squeeze(1)
        next_vectors = torch.FloatTensor(next_vectors).to(device).squeeze(1)
        actions = torch.FloatTensor(actions).to(device)  # actions를 FloatTensor로 변환
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(device).unsqueeze(1)

        # Q(s, a) 계산
        q_values = self.network(images, vectors)

        actions = actions.squeeze(1).long()

        # 각 행동에 대해 one-hot 인코딩 수행
        print("wheel_torque: ", actions[:, 0])
        print("steering_angle: ", actions[:, 1])
        print("brake_state: ", actions[:, 2])
        wheel_torque_one_hot = F.one_hot(actions[:, 0], num_classes=11).to(device)
        steering_angle_one_hot = F.one_hot(actions[:, 1], num_classes=11).to(device)
        brake_state_one_hot = F.one_hot(actions[:, 2], num_classes=2).to(device)

        # one-hot 인코딩된 행동을 결합
        one_hot_action = torch.cat((wheel_torque_one_hot, steering_angle_one_hot, brake_state_one_hot), dim=1).float()
        q_values = torch.sum(q_values * one_hot_action, dim=1).unsqueeze(1)

        # Q(s', a') 계산
        target_q_values = self.target_network(next_images, next_vectors).detach()
        max_q_values = torch.max(target_q_values, dim=1, keepdim=True)[0]
        target_values = rewards + (1 - dones) * discount_factor * max_q_values

        # TD 오차 계산
        td_errors = target_values - q_values
        loss = (td_errors ** 2) * weights
        loss = loss.mean()

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TD 오차로 우선순위 업데이트
        td_errors_np = td_errors.data.cpu().numpy().squeeze()
        td_errors_np = np.abs(td_errors_np) + 1e-6  # NaN 방지 및 소수 값 보정
        self.memory.update_priorities(indices, td_errors_np.flatten())

        # Epsilon 감소
        if self.epsilon > epsilon_min:
            self.epsilon -= epsilon_delta

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
    env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0, target_frame_rate=60)
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    vector_obs = decision_steps.obs[0]
    image_obs = decision_steps.obs[1]
    pos_obs = decision_steps.obs[2]
    print("pos_obs: ", pos_obs.shape)
    print("pos_obs: ", pos_obs)

    vector_size = vector_obs.shape[1]
    image_dim = (image_obs.shape[3], image_obs.shape[1], image_obs.shape[2])  # (channels, height, width)

    print("vector_size: ", vector_size)
    agent = DQNAgent(image_dim, vector_size)

    losses, scores, episode, score = [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0, target_frame_rate=60)
            env.reset()

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(decision_steps) > 0:
            vector = decision_steps.obs[0]  # 벡터 데이터
            image = decision_steps.obs[1]  # 이미지 데이터
        else:
            vector = terminal_steps.obs[0]
            image = terminal_steps.obs[1]

        if len(decision_steps) > 0:  # 에이전트가 존재하는지 확인
            action = agent.get_action(image, vector, train_mode)
            action_tuple = ActionTuple()
            action_tuple.add_discrete(action)  # 이산적 행동 (휠 토크, 조향 각도, 브레이크 상태)

            env.set_actions(behavior_name, action_tuple)
            env.step()

            # 추가된 부분: step 후 결과를 다시 가져오는 코드
            decision_steps, terminal_steps = env.get_steps(behavior_name)
        else:
            env.step()

        done = len(terminal_steps) > 0
        reward = terminal_steps.reward if done else decision_steps.reward
        next_vector = terminal_steps.obs[0] if done else decision_steps.obs[0]
        next_image = terminal_steps.obs[1] if done else decision_steps.obs[1]

        score += reward[0]

        # Debug print statements
        print(f'Step: {step}, Reward: {reward[0]}, Epsilon: {agent.epsilon:.4f}')

        if train_mode:
            agent.append_sample(image, vector, action, reward, next_image, next_vector, [done])

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
                agent.write_summray(mean_score, mean_loss, agent.epsilon, step)
                losses, scores = [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " + \
                      f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

            # 네트워크 모델 저장
            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()
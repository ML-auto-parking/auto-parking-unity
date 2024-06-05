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

# DQN을 위한 파라미터 값 세팅
action_size = 3  # 연속적인 행동 두 개 (휠 토크, 조향 각도) + 이산적인 행동 하나 (브레이크 상태)


load_model = False
train_mode = True

batch_size = 1024
mem_maxlen = 10000
discount_factor = 0.8
learning_rate = 0.00025

run_step = 50000 if train_mode else 0
test_step = 100000
train_start_step = 5000
target_update_step = 500

print_interval = 10
save_interval = 100

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.8
epsilon_delta = (epsilon_init - epsilon_min) / explore_step if train_mode else 0.

# 유니티 환경 경로
game = "AutoParking"
version = 7
env_name = f'../auto-parking-unity-20/Env/ap-{version}'

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/DQN/{date_time}"
load_path = f"./saved_models/{game}/DQN/20210514201212"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN 클래스 -> Deep Q Network 정의
class DQN(torch.nn.Module):
    def __init__(self, image_dim, vector_dim, output_dim):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=image_dim[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        convw = ((image_dim[1] - 8) // 4 + 1 - 4) // 2 + 1 - 3 + 1
        convh = ((image_dim[2] - 8) // 4 + 1 - 4) // 2 + 1 - 3 + 1

        linear_input_size = convw * convh * 64 + vector_dim

        self.fc1 = torch.nn.Linear(linear_input_size, 512)
        self.q = torch.nn.Linear(512, output_dim)

    def forward(self, image, vector):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, vector), dim=1)
        x = F.relu(self.fc1(x))
        return self.q(x)

# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의
class DQNAgent:
    def __init__(self, image_dim, vector_dim):
        self.network = DQN(image_dim, vector_dim, action_size).to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)

        if load_model:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path + '/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def get_action(self, image, vector, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval

        if epsilon > random.random():
            continuous_action = np.random.uniform(-1, 1, size=(vector.shape[0], 2))  # 휠 토크와 조향 각도
            discrete_action = np.random.randint(0, 2, size=(vector.shape[0], 1))  # 브레이크 상태
            action = np.concatenate([continuous_action, discrete_action], axis=1)
        else:
            image_tensor = torch.FloatTensor(image).to(device).permute(0, 3, 1, 2)  # (batch, height, width, channels) -> (batch, channels, height, width)
            vector_tensor = torch.FloatTensor(vector).to(device)
            q = self.network(image_tensor, vector_tensor)
            action = q.data.cpu().numpy()
            # Ensure discrete action is reshaped correctly
            action[:, 2] = np.round(action[:, 2]).astype(int)
            action[:, 2] = action[:, 2].reshape(-1, 1)

        return action

    def append_sample(self, image, vector, action, reward, next_image, next_vector, done):
        self.memory.append((image, vector, action, reward, next_image, next_vector, done))

    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        image = np.stack([b[0] for b in batch], axis=0)
        vector = np.stack([b[1] for b in batch], axis=0)
        action = np.stack([b[2] for b in batch], axis=0)
        reward = np.stack([b[3] for b in batch], axis=0)
        next_image = np.stack([b[4] for b in batch], axis=0)
        next_vector = np.stack([b[5] for b in batch], axis=0)
        done = np.stack([b[6] for b in batch], axis=0)

        image, vector, action, reward, next_image, next_vector, done = map(
            lambda x: torch.FloatTensor(x).to(device), [image, vector, action, reward, next_image, next_vector, done])

        image = image.permute(0, 3, 1, 2)  # (batch, height, width, channels) -> (batch, channels, height, width)
        next_image = next_image.permute(0, 3, 1, 2)

        q = self.network(image, vector)
        q_action = q.gather(1, action.long().unsqueeze(1)).squeeze(1)

        next_q = self.target_network(next_image, next_vector)
        next_q_action = next_q.max(1)[0]

        target = reward + (1 - done) * discount_factor * next_q_action
        loss = F.smooth_l1_loss(q_action, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
    # 유니티 환경 초기화 및 관측 데이터 크기 추출
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=20.0)
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    vector_size = None
    image_dim = None

    # 벡터/이미지 데이터 크기 초기화
    for idx, obs in enumerate(decision_steps.obs):
        if len(obs.shape) == 2:  # 벡터 데이터
            vector_size = obs.shape[1]
        elif len(obs.shape) == 4:  # 이미지 데이터
            image_dim = (obs.shape[3], obs.shape[1], obs.shape[2])  # (channels, height, width)

    agent = DQNAgent(image_dim, vector_size)

    losses, scores, episode, score = [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=5.0)

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
            action_tuple.add_continuous(action[:, :2])  # 연속적 행동 (휠 토크, 조향 각도)
            action_tuple.add_discrete(action[:, 2].reshape(-1, 1).astype(int))  # 이산적 행동 (브레이크 상태)

            env.set_actions(behavior_name, action_tuple)
            env.step()
        else:
            env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        done = len(terminal_steps) > 0
        reward = terminal_steps.reward if done else decision_steps.reward
        next_image = terminal_steps.obs[0] if done else decision_steps.obs[0]
        next_vector = terminal_steps.obs[1] if done else decision_steps.obs[1]
        score += reward[0]

        # 각 스텝마다 reward 출력
        print(f'Step: {step}, Reward: {reward[0]}')

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

            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_loss = np.mean(losses)
                agent.write_summary(mean_score, mean_loss, agent.epsilon, step)
                losses, scores = [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +
                      f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()
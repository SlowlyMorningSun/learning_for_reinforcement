import gym
import sys
import itertools
import matplotlib
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.kernel_approximation import RBFSampler as RBF

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置显示方式
# %matplotlib inline
matplotlib.style.use('ggplot')

env = gym.envs.make("MountainCar-v0")

# 值函数近似器
class EStimator():
    def __init__(self):
        # 对环境进行采样，便于后续对于状态state抽取特征
        state_examples = np.array([env.observation_space.sample() for x in range(10000)])

        # 特征处理1：归一化状态数据为0均值和单位方差
        self.scaler = Scaler()
        self.scaler.fit(state_examples)

        # 特征处理2：状态state的特征抽取表示
        self.featurizer = FeatureUnion([("rbf1", RBF(gamma=5.0, n_components=100)),("rbf2",RBF(gamma=1.0, n_components=100)),])
        self.featurizer.fit(self.scaler.transform(state_examples))

        # 动作空间模型
        # 声明动作模型
        self.action_models = []
        # 动作空间数量
        self.nA = env.action_space.n

        for na in range(self.nA):
            # 动作模型使用随机梯度下降算法
            model = SGD(learning_rate="constant") 
            model.partial_fit([self.featurize_state(env.reset())],[0])
            self.action_models.append(model)

    # 返回状态信号的特征化表示形式
    def featurize_state(self, state):
        # 对输入的状态信号归一化
        scaled = self.scaler.transform([state])
        # 对归一化的状态提取特征
        featurized = self.featurizer.transform(scaled)[0]
        # 返回状态信号的特征化表示形式
        return featurized
    
    # 对状态值函数进行预测
    def predict(self,s):
        # 对输入的状态信号提取特征
        features = self.featurize_state(s)
        # 预测该状态信号对应所有的动作的概率
        predicter = np.array([model.predict([features])[0] for model in self.action_models])

        # 返回动作预测向量
        return predicter

    # 更新值函数近似器
    def update(self, s, a, y):
        # 对当前的状态信号提取特征
        cur_features = self.featurize_state(s)

        # 根据目标y和当前的状态信号特征更新对应的近似模型
        self.action_models[a].partial_fit([cur_features],[y])

# 基于值函数近似表示的时间差分控制的Q-learning算法
class VF_QLearning():
    def __init__(self, env, estimator, num_episodes, epsilon=0.1, discount_factor=1.0, epsilom_decay=1.0):
        
        # 初始化类中的参数
        # 动作空间数量
        self.nA = env.action_space.n
        # 状态空间数量
        self.nS = env.observation_space.shape[0]
        # 环境
        self.env = env
        # 经验轨迹迭代次数
        self.num_episodes = num_episodes
        # epsilon贪婪算法参数
        self.epsilon = epsilon
        # 未来折扣系数
        self.discount_factor = discount_factor
        # 贪婪算法策略衰减系数
        self.epsilon_decay = epsilom_decay
        # 函数近似器
        self.estimator = estimator

        # 记录器，用于保存迭代长度（episode length）和迭代奖励（episode rewards）
        record_head = namedtuple("Stats",["episode_lengths", "episode_rewards"])

        # 记录器初始化
        self.record = record_head(episode_lengths = np.zeros(num_episodes),episode_rewards = np.zeros(num_episodes))

    # epislon 贪婪算法
    def epislon_greedy_policy(self, nA, epislon = 0.5):
        def policy(state):
            A = np.ones(nA, dtype=float) * epislon / nA
            Q = self.estimator.predict(state)
            best_action = np.argmax(Q)
            A[best_action] += (1.0 - epislon)

            return  A
        return  policy

    # 选取随机动作 
    def random_action(self, action_prob):
        
        # 从给定的动作概率action_prob中随机选出一个动作
        return np.random.choice(np.arange(len(action_prob)),p = action_prob)

    # qlearning核心算法
    def q_learning(self):
        for i_episode in range(self.num_episodes):
            # 打印经验轨迹的迭代次数信息
            # 迭代百分比
            num_present = (i_episode + 1)/self.num_episodes
            print("Episode {} / {}".format(i_episode+1,self.num_episodes), end="")
            # 信息输出
            print("="*round(num_present*60))

            # 策略选择使用episilon贪婪算法
            # 策略参数
            policy_epislon = self.epsilon * self.epsilon_decay**i_episode
            # 声明策略
            policy = self.epislon_greedy_policy(self.nA, policy_epislon)
            # 记录奖励
            last_reward = self.record.episode_rewards(i_episode - 1)
            sys.stdout.flush()

            # 重置环境进行第一个动作
            state = env.reset()

            # 下一个动作信号的初始化
            next_action = None

            # 单次经验轨迹的迭代
            for t in itertools.count():
                # 根据策略获得当前状态信号的动作值
                action_probs = policy(state)
                action = self.random_action(action_probs)

                # 向前执行一步
                next_state, reward, done, _ = env.step(action)

                # 更新统计信息
                # 更新信号奖励
                self.record.episode_lengths[i_episode] = t

                # 预测下一时间步的动作值
                # 时间差分更新
                q_values_next = EStimator.predict(next_state)

                # 使用时间差分目标作为预测结果更新函数的近似器
                td_target = reward + self.discount_factor * np.max(q_values_next)
                # Q-Value 时间差分目标
                EStimator.update(state, action, td_target)

                print("\rStep {} with reward ({})".format(t, last_reward),end="")
                if done:break

                # 覆盖下一时间步状态为当前的时间状态
                state = next_state
        return self.record



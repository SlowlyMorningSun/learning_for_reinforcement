import gym
import numpy as np
import pandas as pd
import sys
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt



# CaerPole 游戏介绍
env = gym.make("CartPole-v0")
# sumlist = []
# for t in range(200):
#     state = env.reset()
#     i = 0

#     # 进行游戏
#     while(True):
#         i +=1
#         # 环境重置
#         env.render()
#         # 随机选择动作
#         action = env.action_space.sample()
#         # 获取动作的数量
#         nA = env.action_space.n
#         # 智能体执行动作
#         state, reward, done, _ = env.step(action)
#         # print(state,action ,reward)

#         # 游戏结束，输出本次游戏的时间步
#         if done:
#             print("Episode finished after {} timesteps".format(i+1))
#             break
#     # 记录迭代次数
#     sumlist.append(i)
#     print("Gamw Over .....")
# # 关闭游戏监听器
# env.close()

# iter_time = sum(sumlist)/len(sumlist)
# print("CartPole game iter average time is {}".format(iter_time))

# 显示时间步和奖励结果
def plot_episodes_stats(stats, smoothing_window=10):
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths[:200])
    plt.xlabel("Episode")
    plt.ylabel("Episode length")
    plt .title("EPisode Length over time")
    plt.show(fig1)
    fig2 = plt.figure(figsize=(10,5))
    reward_smoothed = pd.Series(stats.episode_rewards[:200]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(reward_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("Episode Reward over time".format(smoothing_window))
    plt.show(fig2)
    return fig1, fig2



# class SARSA():
#     def __init__(self, env, num_episodes, discount=1.0, alpha=0.5, epsilon=0.1, n_bins=10):
#         # 初始化算法使用到的基本变量
#         # 动作状态数
#         self.nA = env.action_space.n
#         # 状态空间数
#         self.nS = env.observation_space.shape[0]
#         # 环境
#         self.env = env
#         # 迭代次数
#         self.num_episodes = num_episodes
#         # 衰减系数
#         self.discount = discount
#         # 时间差分误差系数
#         self.alpha = alpha
#         # 贪婪策略系数
#         self.epsilon = epsilon
#         # 动作值函数
#         self.Q = defaultdict(lambda: np.zeros(self.nA))

#         # 记录重要的迭代信息
#         record = namedtuple("Record",["episode_lengths","episode_rewards"])
#         self.rec = record(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))

#         # 状态空间的桶
#         self.cart_position_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1]
#         self.pole_angle_bins = pd.cut([-2.0, 2.0], bins=n_bins, retbins=True)[1]
#         self.cart_velocity_bins = pd.cut([-1.0, 1.0], bins=n_bins, retbins=True)[1]
#         self.angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins, retbins=True)[1]
    
#     # 状态空间简化后的返回函数 
#     def get_bins_states(self, state):

#         # 获取当前状态的4个状态元素值
#         s1_, s2_, s3_, s4_ = state

#         # 分别找到4个元素值在bins中的索引位置
#         cart_position_idx = np.digitize(s1_, self.cart_position_bins)
#         pole_angle_idx = np.digitize(s2_, self.pole_angle_bins)
#         cart_velocity_idx = np.digitize(s3_, self.cart_velocity_bins)
#         angle_rate_idx = np.digitize(s4_, self.angle_rate_bins)

#         # 重新组合简化过的状态值
#         state_ = [cart_position_idx, pole_angle_idx, cart_velocity_idx, angle_rate_idx]

#         # 通过map函数对状态索引号进行组合，并把每一个元素强制转换为int类型
#         state = map(lambda s :int(s), state_)
#         return tuple(state)

#     # epislon 贪婪策略
#     def __epislon_greedy_policy(self, epsilon, nA):

#         def policy(state):
#             A = np.ones(nA, dtype=float) * epsilon /nA
#             best_action = np.argmax(self.Q[state])
#             A[best_action] += (1.0 - epsilon)
#             return A
#         return policy

#     # 随机选择动作
#     def __next_action(self, prob):
#         return np.random.choice(np.arange(len(prob)), p=prob)

#     # sarsa算法核心流程代码
#     def sarsa(self):
#         # Sarsa 算法
#         policy = self.__epislon_greedy_policy(self.epsilon, self.nA)
#         sumlist = []

#         # 迭代经验轨迹
#         for i_episodes in range(self.num_episodes):
#             # 输出迭代的信息
#             if 0 == (i_episodes+1) % 10:
#                 print("\r Episode {} in {}".format(i_episodes+1, self.num_episodes))
            
#             # 每一次迭代的初始化状态s，动作状态转换概率p，下一个动作a
#             step = 0
#             # 初始化状态
#             state__ = self.env.reset()
#             # 状态重新赋值
#             state = self.get_bins_states(state__)
#             # 根据状态获得动作状态的转换概率
#             prob_actions = policy(state)
#             # 选择一个动作
#             action = self.__next_action(prob_actions)
#             # 迭代本次经验轨迹
#             while(True):
#                 next_state__, reward, done, info = env.step(action)
#                 next_state = self.get_bins_states(next_state__)

#                 prob_next_actions = policy(next_state)
#                 next_action = self.__next_action(prob_next_actions)

#                 # 更新需要的记录的信息（迭代时间步长和奖励）
#                 self.rec.episode_lengths[i_episodes] += reward
#                 self.rec.episode_rewards[i_episodes] = step

#                 # 时间差分更新
#                 td_target = reward + self.discount * self.Q[next_state][next_action] 
#                 td_delta = td_target - self.Q[state][action]
#                 self.Q[state][action] += self.alpha * td_delta

#                 if done:
#                     # 游戏结束
#                     reward = -200
#                     print("Episode finished after {} timesteps".format(step))
#                     sumlist.append(step)
#                     break
#                 else:
#                     # 状态和动作重新赋值
#                     step += 1
#                     state = next_state
#                     action = next_action
#         # 结束本次经验轨迹之前进行平均奖励得分统计，并输出结果
#         iter_time = sum(sumlist)/len(sumlist)
#         print("CartPole game iter average time is {}".format(iter_time))
#         return self.Q
# cls_sarsa = SARSA(env,num_episodes=1000)
# Q = cls_sarsa.sarsa()
# plot_episodes_stats(cls_sarsa.rec)

# Q-learing 算法
class QLearning():
    def __init__(self, env, num_episodes, discount=1.0, alpha=0.5, epsilon=0.1, n_bins=10):
        
        # 动作空间数
        self.nA = env.action_space.n
        # 状态空间数
        self.nS = env.observation_space.shape[0]
        # 环境
        self.env = env
        # 迭代次数
        self.num_episodes = num_episodes
        # 衰减系数
        self.discount = discount
        # 时间差分误差系数
        self.alpha = alpha
        # 贪婪策略系数
        self.epsilon = epsilon

        # 初始化动作值函数
        # Initialize Q(s,a)
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        # 定义存储记录有用的信息(每一条经验轨迹的时间步与奖励)
        # keeps track of useful statistics
        record = namedtuple("Record",["episode_lengths","episode_rewards"])
        self.rec = record(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))

        # 状态空间的桶
        self.cart_position_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1]
        self.pole_angle_bins = pd.cut([-2.0, 2.0], bins=n_bins, retbins=True)[1]
        self.cart_velocity_bins = pd.cut([-1.0, 1.0], bins=n_bins, retbins=True)[1]
        self.angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins, retbins=True)[1]

    # 状态空间简化后的返回函数 
    def get_bins_states(self, state):

        # 获取当前状态的4个状态元素值
        s1_, s2_, s3_, s4_ = state

        # 分别找到4个元素值在bins中的索引位置
        cart_position_idx = np.digitize(s1_, self.cart_position_bins)
        pole_angle_idx = np.digitize(s2_, self.pole_angle_bins)
        cart_velocity_idx = np.digitize(s3_, self.cart_velocity_bins)
        angle_rate_idx = np.digitize(s4_, self.angle_rate_bins)

        # 重新组合简化过的状态值
        state_ = [cart_position_idx, pole_angle_idx, cart_velocity_idx, angle_rate_idx]

        # 通过map函数对状态索引号进行组合，并把每一个元素强制转换为int类型
        state = map(lambda s :int(s), state_)
        return tuple(state)

    # epislon 贪婪策略
    def __epislon_greedy_policy(self, epsilon, nA):

        def policy(state):
            A = np.ones(nA, dtype=float) * epsilon /nA
            best_action = np.argmax(self.Q[state])
            A[best_action] += (1.0 - epsilon)
            return A
        return policy

    # 随机选择动作
    def __next_action(self, prob):
        return np.random.choice(np.arange(len(prob)), p=prob)

    # Q-learning 算法
    def qlearning(self):
        # 定义策略
        policy = self.__epislon_greedy_policy(self.epsilon, self.nA)
        sumlist = []

        # 开始迭代经验轨迹
        for i_episode in range(self.num_episodes):
            if 0 == (i_episode + 1) % 10:
                print("\r Episode {} in {}".format(i_episode+1, self.num_episodes))
            
            # 初始化环境状态并对状态值进行索引简化
            step = 0
            state__ = self.env.reset()
            state = self.get_bins_states(state__)

            # 迭代本次经验轨迹
            while(True):
                # 根据策略，在状态s下选择动作a
                prob_actions = policy(state)
                action = self.__next_action(prob_actions)

                # 智能体执行动作
                next_state__, reward, done, info = env.step(action)
                next_state = self.get_bins_states(next_state__)
                
                # 更新每一条经验轨迹的时间步与奖励
                self.rec.episode_lengths[i_episode] += reward
                self.rec.episode_rewards[i_episode] = step

                # 更新动作值函数
                # Q（S，A） <-- Q(S,A) + alpha * [R + discount * max Q(S';a) - Q(S,A)]
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.discount * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta 

                if done:
                    # 游戏停止，输出输出结果
                    print("Episode finished after {} timesteps".format(step))
                    sumlist.append(step)
                    break
                else:
                    # 游戏继续，状态赋值
                    step += 1
                    # S <- S'
                    state = next_state
        # 结束本次经验轨迹之前进行平均奖励得分统计，并输出结果
        iter_time = sum(sumlist)/len(sumlist)
        print("CartPole game iter average time is {}".format(iter_time))
        return self.Q
cls_qlearning = QLearning(env,num_episodes=1000)
Q = cls_qlearning.qlearning()
plot_episodes_stats(cls_qlearning.rec)

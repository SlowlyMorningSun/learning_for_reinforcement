import gym
import sys
import itertools
import matplotlib
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

env = gym.envs.make('CartPole-v0')
env = env.unwrapped
env.seed(1)
# 蒙特卡洛策略梯度类
class Monte_Carlo_Policy_Gradient():
    # 蒙特卡洛策略梯度方法类
    def __init__(self, env, num_episodes=200, learning_rate = 0.01, reward_decay = 0.95):
        # 初始化参数
        # 动作空间
        self.nA = env.action_space.n
        # 状态空间
        self.nS = env.observation_space.shape[0]
        # 声明环境
        self.env = env
        # 迭代次数
        self.num_episodes = num_episodes
        # 奖励衰减系数
        self.reward_decay = reward_decay
        # 网络学习率
        self.learning_rate = learning_rate
        # 记录所有的奖励
        self.rewards = []
        # 最小奖励阈值
        self.RENDER_REWARD_MIN = 30
        # 是否重新分配环境标志位
        self.RENDER_ENV = False

        # 初始化策略网络类
        self.PG = PolicyGradient(n_x=self.nS, n_y = self.nA, learning_rate = self.learning_rate, reward_decay = self.reward_decay)

        # 记录经验轨迹的长度和奖励
        record_head = namedtuple("Stats", ["episode_lengths","episode_rewards"])
        self.record = record_head(episode_lengths = np.zeros(num_episodes), episode_rewards = np.zeros(num_episodes))
    
    # 蒙特卡洛策略梯度算法
    def mcpg_learn(self):
        # 迭代经验轨迹的次数
        for i_episode in range(self.num_episodes):
            # 输出经验轨迹迭代信息
            num_present = (i_episode+1) / self.num_episodes
            print("Episode {} / {}".format(i_episode+1,self.num_episodes), end="")
            # 信息输出
            print("="*round(num_present*60))

            # 初始化环境
            # 环境reset
            state = env.reset()
            
            # 初始化奖励为0
            reward = 0

            # 遍历经验轨迹
            for t in itertools.count():
                # 如果环境重置标志位为True，则对环境重置
                if self.RENDER_ENV: env.render()

                # 步骤1：跟据给定的状态，策略网络选择出相应的动作
                action = self.PG.choose_action(state)
                # 步骤2：环境执行动作给出反馈信号
                next_state, reward, done, _ = env.step(action)
                # 步骤3：记录环境反馈信号，用于策略网络的训练数据
                self.PG.store_memory(state, action, reward)

                # 更新记录的信息
                self.record.episode_rewards[i_episode] += reward
                self.record.episode_lengths[i_episode] = t

                # 游戏结束
                if done:
                    # 计算本次经验轨迹所获得的累计奖励
                    episode_rewards_sum = sum(self.PG.episode_rewards)
                    self.rewards.append(episode_rewards_sum)
                    max_reward = np.max(self.rewards)

                    # 步骤4：结束游戏后对策略网络进行训练
                    self.PG.learn()

                    # 标准化输出信息
                    print("reward:{},max reward:{}, episode:{}\n".format(episode_rewards_sum, max_reward, t))

                    # 如果历史最大奖励大于奖励的最小阈值，则重置环境标志位为True
                    if max_reward > self.RENDER_REWARD_MIN : self.RENDER_ENV = True

                    # 退出本次经验轨迹
                    break
                # 步骤5：存储下一个状态作为新的状态记录
                state = next_state
        # 返回记录数据
        return self.record            

class PolicyGradient():
    # 策略梯度强化学习类，使用一个3层的神经网络作为策略网络
    def __init__(self, n_x, n_y, learning_rate=0.01, reward_decay=0.95, load_path=None, save_path=None):
        # 策略梯度类构造函数，初始化相关参数

        # 初始化参数
        # 策略网络输入
        self.n_x = n_x
        # 策略网络输出
        self.n_y = n_y
        # 策略网络学习率
        self.lr = learning_rate
        # 策略网络奖励衰减率
        self.reward_decay = reward_decay

        # 经验轨迹采样数据（s，a，r）
        self.episode_states, self.episode_actions, self.episode_rewards = [],[],[]
        # 建立策略网络
        self.__build_network()
        
    def all_actf(self):
        all_act = self.dense_out
        # print(all_act)
        return all_act
    def reca_batch(self,a_batch):
        a = a_batch
        return a
    def def_loss(self,label=reca_batch,logit=all_actf):  
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logit)
        # loss = tf.reduce_mean(neg_log_prob * self.disc_norm_ep_reward)
        # return loss
        return neg_log_prob

    def __build_network(self):
        # 建立一个三层的神经网络
        # 输入层
        x = Input(shape= (self.n_x,))
        # 定义层数的神经元
        # 第一层的隐层神经元数
        layer1_units = 10
        # 第二层隐层神经元元数
        layer2_unit = 10
        # 输出层的神经元数
        layer_output_units = self.n_y
        # 定义第一层
        dense1 = Dense(layer1_units, activation = 'relu', name='layer1')(x)
        # 定义第二层
        dense2 = Dense(layer2_unit, activation= 'relu', name='layer2')(dense1)
        # 定义输出层
        self.dense_out = Dense(layer_output_units,name='out_layer')(dense2)
        # softmax 输出
        self.outputs_softmax = Softmax()(self.dense_out)

        self.model = Model(x,self.outputs_softmax)

        # # 定义loss函数
        # def ccloss(label=self.episode_actions,logit=self.dense_out):
        #     # print(logit)
        #     # print(label)
            
        #     neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label)
        #     # print(neg_log_prob)
        #     print(self.disc_norm_ep_reward)
        #     loss = tf.reduce_mean(neg_log_prob * self.disc_norm_ep_reward)
        #     return loss

        # 定义神经网络的损失函数与训练方式
        self.model.compile(optimizer=Adam(learning_rate=self.lr),loss=self.def_loss)

        self.model.summary()

    def __disc_and_norm_rewards(self):
        out = np.zeros_like(self.episode_rewards)
        dis_reward = 0

        for i in reversed(range(len(self.episode_rewards))):
            dis_reward =  self.reward_decay * dis_reward +  self.episode_rewards[i]  # 前一步的reward等于后一步衰减reward加上即时奖励乘以衰减因子
            out[i] = dis_reward
        return  out/np.std(out - np.mean(out))

    # 根据给定的状态选择对应的动作
    def choose_action(self, state):
        # 对状态的存储格式进行转换，便于神经网络的输入
        state = state[ np.newaxis,:]
        # 神经网络的前馈计算
        # prob_actions = self.sess.run(self.outputs_softmax, feed_dict={self.X: state})
        prob_actions = (self.model.predict(state))
        
        # print(prob_actions.ravel())
        # 根据得到的动作概率随机选择一个作为需要执行的动作
        action = np.random.choice(range(len(prob_actions.ravel())),p=prob_actions.ravel())
        return action
    
    # 存储经验轨迹产生的数局作为后续神经网络的训练数据
    def store_memory(self, state, action, reward):

        # 记录状态数据
        self.episode_states.append(state)
        # 记录奖励数据
        self.episode_rewards.append(reward)
        # 创建动作空间
        action__ = np.zeros(self.n_y)
        # 当执行的动作设置为1其余为0
        action__[action] = 1
        # 创建动作空间
        self.episode_actions.append(action__)

    # 根据经验轨迹数据对神经网络进行训练
    def learn(self):
        # 奖励数据处理
        self.disc_norm_ep_reward = self.__disc_and_norm_rewards()
        # print(self.disc_norm_ep_reward)
        # self.sess.run(self.trian_op, feed_dict={self.X: np.vstack(self.episode_states).T, self.Y: np.vstack(self.episode_actions).T, self.disc_norm_ep_reward: disc_norm_ep_reward,})
        # print("X,",np.vstack(self.episode_states))
        # print("Y,",np.vstack(self.episode_actions))
        
        self.model.fit(np.vstack(self.episode_states),np.array(self.episode_actions),sample_weight=self.disc_norm_ep_reward)
        # 重置经验轨迹数据用于记录下一条经验轨迹
        self.episode_states, self.episode_actions, self.episode_rewards = [],[],[]

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

mcpg = Monte_Carlo_Policy_Gradient(env, num_episodes=200)
result = mcpg.mcpg_learn()
plot_episodes_stats(result)

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
# 演员-评论家策略梯度类
class Actor_Critic():
    # 演员-评论家策略梯度方法类
    def __init__(self, env, num_episodes=200, learning_rate = 0.01, reward_decay = 0.95):
        # 初始化参数
        # 动作空间
        self.nA = env.action_space.n
        # 状态空间
        self.nS = env.observation_space.shape[0]
        # 
        self.nR = 1
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
        self.Actor = PolicyGradient(n_x=self.nS, n_y = self.nA, learning_rate = self.learning_rate, reward_decay = self.reward_decay)
        self.Critic = ValueEstimator(n_x=self.nS, n_y = self.nR, learning_rate = self.learning_rate)
        # 记录经验轨迹的长度和奖励
        record_head = namedtuple("Stats", ["episode_lengths","episode_rewards"])
        self.record = record_head(episode_lengths = np.zeros(num_episodes), episode_rewards = np.zeros(num_episodes))
    
    # 演员-评论家策略梯度算法核心代码
    def ac_learn(self):
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
            reward_ = 0

            # 遍历经验轨迹
            for t in itertools.count():
                
                # 如果环境重置标志位为True，则对环境重置
                if self.RENDER_ENV: env.render()

                # 步骤1：根据策略网络（Actor网络）选择出相应的动作
                action = self.Actor.choose_action(state)
                # 步骤2：环境执行动作给出反馈信号
                next_state, reward, done, _ = env.step(action)
                print("state:", next_state,"reward:",reward, "action:",action)
                # 更新奖励
                reward_ +=reward 

                # 步骤3：记录时间差分误差
                # 评论家预测的下一个状态价值
                value_next = self.Critic.predict(next_state)
                # 计算时间差分目标
                td_target = reward + self.reward_decay * value_next
                # 计算时间差分误差
                td_error = td_target - self.Critic.predict(state)
                # self.Actor.store_transition(td_error)
                print("value_next:",value_next,"TD_targrt:",td_target)
                # 步骤4：更新价值网络（评议家网络）
                self.Critic.learn(state, td_target)
                # 步骤5：更新策略网络（演员网络）
                self.Actor.store_memory(state, action, td_error)
                self.Actor.learn()

                # 更新记录的信息
                self.record.episode_rewards[i_episode] += reward
                self.record.episode_lengths[i_episode] = t

                # 游戏结束
                if done:
                    # 计算本次经验轨迹所获得的累计奖励
                    self.rewards.append(reward_)
                    # 
                    max_reward = np.max(self.rewards)

                    # 历史最大奖励信息，标准化输出信息
                    print("reward:{},max reward:{}, episode:{}\n".format(reward_, max_reward, t))

                    # 如果历史最大奖励大于奖励的最小阈值，则重置环境标志位为True
                    if max_reward > self.RENDER_REWARD_MIN : self.RENDER_ENV = True

                    # 退出本次经验轨迹
                    break
                # 步骤5：存储下一个状态作为新的状态记录
                state = next_state
        # 返回记录数据
        return self.record            

# 演员网络部分（Actor）
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
        # 经验轨迹采样数据（s，a，td_error）
        self.episode_states, self.episode_actions, self.episode_tderrors = [],[],[]
        # 建立策略网络
        self.__build_network()

    # 存储经验轨迹产生的数局作为后续神经网络的训练数据
    def store_memory(self, state, action, td_error):

        # 记录状态数据
        self.episode_states.append(state)
        # 记录奖励数据
        self.episode_tderrors.append(td_error)
        # 创建动作空间
        action__ = np.zeros(self.n_y)
        # 当执行的动作设置为1其余为0
        action__[action] = 1
        # 创建动作空间
        self.episode_actions.append(action__)

    # loss的定义   
    def all_actf(self):
        all_act = self.dense_out
        # print(all_act)
        return all_act
    def reca_batch(self,a_batch):
        a = a_batch
        return a
    def def_loss(self,label=reca_batch,logit=all_actf):  
        neg_log_prob = tf.math.squared_difference(logit,label)
        loss = tf.reduce_mean(neg_log_prob, self.episode_tderrors[0])
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

        # 定义神经网络的损失函数与训练方式
        self.model.compile(optimizer=Adam(learning_rate=self.lr),loss=self.def_loss)

        self.model.summary()

    # 根据给定的状态选择对应的动作
    def choose_action(self, state):
        # 获得action的索引值
        state = state[ np.newaxis,:]
        # 神经网络的前馈计算
        prob_actions = (self.model.predict(state))
        
        # 根据得到的动作概率随机选择一个作为需要执行的动作
        action = np.random.choice(range(len(prob_actions.ravel())),p=prob_actions.ravel())
        return action
    

    # 根据经验轨迹数据对神经网络进行训练
    def learn(self):
        
        
        # print(np.array(state))
        # print(np.array(action__))
        # print(state.shape,action__.shape)
        # print(action__)
        self.model.fit([np.vstack(self.episode_states)],[np.array(self.episode_actions)])
        self.episode_states, self.episode_actions, self.episode_tderrors = [],[],[]

# 评论家部分（Critic）
class ValueEstimator():
    def __init__(self, n_x ,n_y, learning_rate = 0.01, gama = 0.95 ,Load_path = None, Save_path=None):
        
        # 状态的尺寸
        self.nS = n_x
        # 动作值的尺寸
        self.nR = n_y
        # 学习率
        self.lr = learning_rate
        #
        self.gama = 0.95
        # 建立网络
        self.critic_network()
   
    # loss的定义   
    def all_actf(self):
        all_act = self.dense_out
        # print(all_act)
        return all_act
    def reca_batch(self,a_batch):
        a = a_batch
        return a
    # label 为 真实 td_error         logit 为 self.denseout 的输出值
    # r + gama * v_next - v
    def def_loss(self,label=reca_batch,logit=all_actf):  
        # loss = self.reward_now + self.gama * self.v_next - logit
        # loss = tf.square(loss)
        # return loss
        neg_log_prob = tf.math.squared_difference(logit,label)
        return neg_log_prob

    # 建立神经网络
    def critic_network(self):
        x = Input(shape= (self.nS,))
        # 定义层数的神经元
        # 第一层的隐层神经元数
        layer1_units = 20
        # 输出层的神经元数
        layer_output_units = self.nR
        # 定义第一层
        dense1 = Dense(layer1_units, activation = 'relu', name='layer1')(x)
        # 定义输出层（输出的是td_error值）
        self.dense_out = Dense(layer_output_units, name = "output_layer")(dense1)
        # 建立模型
        self.model = Model(x,self.dense_out)

        # 定义神经网络的损失函数与训练方式
        self.model.compile(optimizer=Adam(learning_rate=self.lr),loss=self.def_loss)

        self.model.summary()
    
    # 对状态值函数进行预测
    def predict(self,s):
        
        s = s[np.newaxis,:]
        # 预测该状态函数
        prob_weights = self.model.predict(s)
        # 返回状态函数的值
        return prob_weights[0]

    # 对神经网络进行训练
    def learn(self, s, td_target ):
        # 创建动作空间
        s = s[np.newaxis, :]
        
        self.model.fit(np.vstack(s),np.array(td_target))
        

# 结果画图
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

ac = Actor_Critic(env, num_episodes=200)
result = ac.ac_learn()
plot_episodes_stats(result)

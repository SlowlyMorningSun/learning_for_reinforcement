import gym
import numpy as np
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 21 点的游戏
def simple_strategy(state):
    # 21 点游戏使用的游戏策略；
    # state ： 输入的游戏状态

    # 获取玩家，庄家在游戏中的状态
    player, dealer, ace = state
    return 0 if player >= 18 else 1

# 21 点游戏的经验轨迹收集

# 定义gym环境为Blackjack游戏
env = gym.make("Blackjack-v0")
def show_state(state):
# 用于输出任务的当前状态，玩家点数，庄家点数以及是否持有牌A
# state：输入的状态
    player, dealer, ace = state
    dealer = sum(env.dealer)
    print("Player:{}, ace:{}, Dealer:{}".format(player,ace,dealer))

def episode(num_episodes):
# 收集经验轨迹函数
# num_episodes : 迭代次数
    episode = [] # 经验轨迹收集列表

    # 迭代num_episodes条经验轨迹
    for i_episode in range(num_episodes):
        print("\n"+"="* 30)
        state = env.reset()

        # 每条经验轨迹有10个状态
        for t in range(10):

            show_state(state)
            # 基于某一个策略选择动作
            action = simple_strategy(state)
            # 对于玩家Player 只有Stand 停牌，和HIT拿牌两种动作
            action_ = ["STAND","HIT"][action]
            print("Player Simple Strategy take action:{}".format(action_))

            # 执行某一策略下的动作
            next_state, reward, done, _ = env.step(action)

            # 记录经验轨迹
            episode.append((state,action, reward))

            # 遇到游戏结束结束打印游戏结果
            if done:
                show_state(state)
                # [-1(loss),-(push), 1(win)]
                reward_ = ["loss", "push", "win"][int(reward+1)]
                print("Game {}.(Reward{})".format(reward_,int(reward)))
                print("PLAYER:{}\t DEALER:{}".format(env.player,env.dealer))
                break
            state = next_state

# 首次访问蒙特卡洛预测算法
def mc_firstvisit_prediction(policy, env, num_episodes, episode_endtime=10, discount=1.0):
    # sum记录
    r_sum = defaultdict(float)
    # count记录
    r_count = defaultdict(float)
    # 状态值记录
    r_V = defaultdict(float)

    # 采集num_episodes条经验轨迹
    for i in range(num_episodes):
        # 输出经验轨迹完成的百分比
        episode_rate = int(40 * i / num_episodes)
        print("Episode {}/{}".format(i+1, num_episodes) + "=" * episode_rate,end="\r")
        sys.stdout.flush()

        # 初始化经验轨迹集合和环境状态
        episode = []
        state = env.reset()

        # 完成一条经验轨迹
        for j in range(episode_endtime):
            # 根据给定的策略选择动作，即a = policy（s）
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state,action, reward))

            # 遇到游戏结束结束打印游戏结果
            if done:
                break
            state = next_state

        # 首次访问蒙特卡洛预测的核心算法
        for k, data_k in enumerate(episode):
            # 获得首次遇到该状态的引索号k
            state_k = data_k[0]
            # 计算首次访问的状态的累积奖励
            G = sum([x[2] * np.power(discount,i) for i,x in enumerate(episode[k:])]) # 每次访问蒙特卡洛预测算法，在这里不同
            r_sum[state_k] += G
            r_count[state_k] += 1.0
            # 计算状态值
            r_V[state_k] = r_sum[state_k] / r_count[state_k]
    return r_V

# 构筑三维图像
def plot_value_function(v, title=None):
    x = []
    y = []
    z = []
    for key , values in v.items():
        x.append(key[1])
        y.append(key[0])
        z.append(values)
    fig = plt.figure()  #定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')   # 建立坐标轴  
    #作图
    ax3.plot_trisurf(x, y, z, cmap="rainbow" )
    plt.show()

# 每次访问蒙特卡洛预测算法
def mc_everyvisit_prediction(policy, env, num_episodes, episode_endtime=10, discount=1.0):
    # sum记录
    r_sum = defaultdict(float)
    # count记录
    r_count = defaultdict(float)
    # 状态值记录
    r_V = defaultdict(float)

    # 采集num_episodes条经验轨迹
    for i in range(num_episodes):
        # 输出经验轨迹完成的百分比
        episode_rate = int(40 * i / num_episodes)
        print("Episode {}/{}".format(i+1, num_episodes) + "=" * episode_rate,end="\r")
        sys.stdout.flush()

        # 初始化经验轨迹集合和环境状态
        episode = []
        state = env.reset()

        # 完成一条经验轨迹
        for j in range(episode_endtime):
            # 根据给定的策略选择动作，即a = policy（s）
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state,action, reward))

            # 遇到游戏结束结束打印游戏结果
            if done:
                break
            state = next_state
        # 每次访问蒙特卡洛预测的核心算法
        for k,data_k in enumerate(episode):
            # 获得首次遇到该状态的引索号k
            state_k = data_k[0]
            # 计算每次访问状态的累计奖励
            G = sum([x[2] * np.power(discount,i) for i,x in enumerate(episode)]) # 首次访问蒙特卡洛预测算法，在这里不同
            r_sum[state_k] += G
            r_count[state_k] += 1.0
            r_V[state_k] = r_sum[state_k]/r_count[state_k]
    return r_V

# epsilon_贪婪策略算法
def epsilon_greddy_policy(q, epsilon, nA):
    def __policy__(state):
        # 初始化动作概率
        A_ = np.ones(nA, dtype=float)

        # 以epsilon设定动作概略
        A = A_ * epsilon / nA

        # 选取动作值函数中的最大值作为最优值
        best = np.argmax(q[state])

        # 以1-epsilon 设定最大动作动作概率
        A[best] += 1- epsilon 
        return A
    return __policy__

# 固定策略的非起始点探索的蒙特卡洛控制
def mc_firstvisit_control_epsilon_greddy(env, num_episodes=100, epsilon=0.1, episode_endtime=10, discount=1.0 ):
    # 初始化设定使用到的变量

    # 环境中的状态对应动作空间数量
    nA = env.action_space.n
    # 动作值函数
    Q = defaultdict(lambda: np.zeros(nA))
    # 动作-状态对的累计奖励
    r_sum = defaultdict(float)
    # 动作-状态对的计数器
    r_cou = defaultdict(float)

    # 初始化贪婪策略
    policy = epsilon_greddy_policy(Q, epsilon, nA)

    for i in range(num_episodes):
        # 输出当前迭代的经验轨迹次数
        episode_rate = int(40 * i / num_episodes)
        print("Episode {}/{}".format(i+1, num_episodes),end = "\r")
        sys.stdout.flush()

        # 初始化状态和当前的经验轨迹
        episode = []
        state = env.reset()

        # (a) 基于策略产生一条经验轨迹，其中每一个事件步为tuple(state, action, reward)
        for j in range(episode_endtime):
            
            # 通过epslion-greddy算法对动作-状态对进行探索和利用
            action_prob = policy(state)
            # 根据epslion-greddy算法的结果随机选取一个动作
            action = np.random.choice(np.arange(action_prob.shape[0]), p=action_prob)

            # 运行一个时间步并采集经验轨迹
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        # (b) 计算经验轨迹中每一个<状态-动作>对
        for k,(state, actions, reward) in  enumerate(episode):

            # 提取动作-状态对为 sa_pair
            sa_pair = (state, actions)
            first_visit_idx = k

            # 计算未来累计奖励
            G = sum([x[2] * np.power(discount, i) for i , x in enumerate(episode[first_visit_idx:])])

            # 更新未来累计奖励
            r_sum[sa_pair] += G
            # 更新动作-状态对的计数器
            r_cou[sa_pair] +=1.0

            # 计算平均的累计奖励
            Q[state][actions]  = r_sum[sa_pair] / r_cou[sa_pair]
    
    return Q

#######################################################################
# 非固定策略学习算法（蒙特卡洛）
# 辅助函数
def create_random_policy(nA):
    # 创建一个随机策略
    A = np.ones(nA, dtype=float) / nA

    # 策略函数
    def policy_fn(observation):
        return A
    
    return policy_fn

def create_greedy_policy(Q):
    # 创建一个贪婪策略

    # 策略函数
    def policy_fn(state):
        A = np.zeros_like(Q[state],dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    
    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    # 非固定式策略学习法

    # 初始化参数
    Q = defaultdict(lambda: np.zeros(env.action_space.n)) # 动作值函数
    C = defaultdict(lambda: np.zeros(env.action_space.n)) # 重要性参数

    # 初始化目标策略
    target_policy = create_greedy_policy(Q)

    # Repect
    for i_episode in range(1,num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode,num_episodes), end="")
            sys.stdout.flush()
        
        # 单条经验轨迹采样,其中每个时间步内容为 turple(state, action, reward)
        episode = []
        state = env.reset()

        while(True):
            # 从行为策略中进行采样
            probs = behavior_policy(state)

            # 随机在当前状态的动作概率中选择一个动作
            action = np.random.choice(np.arange(len(probs)),p=probs)
            
            # 智能体执行该动作并记录当前状态
            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        G = 0.0 # 未来折扣累计奖励
        W = 1.0 # 重要性权重参数

        # 在该经验轨迹中从最后的一个时间步开始遍历
        for t in range(len(episode))[::-1]:

            # 获得当前经验轨迹的当前时间步
            state, action, reward = episode[t]
            # 更新累计奖励
            G = discount_factor * G + reward
            # 求累计权重Cn
            C[state][action] += W
            # 更新动作值函数，同样的，这是改进目标中用到的动作值函数
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            # 如果行为策略采取的动作并不是目标策略的动作，那么概率将变化为0，循环中断
            if action != np.argmax(target_policy(state)):
                break 
            W = W * 1.0 / behavior_policy(state)[action]
    return Q, target_policy
# # 测试每次访问和首次访问蒙特卡洛预测方法的区别    
# v = mc_everyvisit_prediction(simple_strategy, env ,100000)
# # print(v)
# plot_value_function(v)
# # episode(100)

# # 非起始点探索获得动作值函数
# Q = mc_firstvisit_control_epsilon_greddy(env,num_episodes=100000)
# print(Q)
# # 初始化状态函数
# V = defaultdict(float)
# # (c) 根据求得的动作值函数选择最大的动作作为最优状态值
# for state, actions in Q.items():
#     V[state] = np.max(actions)
# plot_value_function(V)

# 非固定式策略学习法
random_policy = create_random_policy(env.action_space.n)
Q, policy =mc_control_importance_sampling(env, num_episodes=500,behavior_policy=random_policy)
V = defaultdict(float)
for state, action_values in Q.items():
    action_values = np.max(action_values)
    V[state] = action_values
plot_value_function(V)



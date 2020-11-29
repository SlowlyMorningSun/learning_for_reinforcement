import numpy as np
import sys
from six import StringIO,b
from gym import utils
from gym.envs.toy_text import discrete


# HelloGrid 环境
MAPS = {'4x4':["SOOO","OXOX","OOOX","XOOG"]}
# 对应所有状态都有四个动作
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


# 关于 HelloGrid 的环境
class HelloGridEnv(discrete.DiscreteEnv):
    metadata = {'render.modes':['human','ansi']}
    
    def __init__(self, desc = None, map_name = '4x4'):
        """ GridWorldEnv 环境构建"""
        # 环境地图Grid
        self.desc = np.asarray(MAPS[map_name],dtype='c')
        # 获取maps的形状（4，4）
        self.shape = self.desc.shape
        # 动作集的个数
        nA = 4
        # 状态集的个数
        nS = np.prod(self.desc.shape)
        # 设置最大的行号和最大的列号方便索引
        MAX_Y = self.shape[0]
        MAX_X = self.shape[1]

        # 初始状态分布[1. 0. 0.  ...],并于格子S开始执行
        isd = np.array(self.desc == b'S').astype('float64').ravel()  # 将其变为对应的一维数组
        isd /= isd.sum()
        # 动作-状态转化概率字典
        P = {}

        # 使用 numpy 的 nditer 时状态 grid 进行遍历
        state_grid = np.arange(nS).reshape(self.desc.shape)
        it = np.nditer(state_grid,flags = ['multi_index'])

        # 通常it.finish,it.iternext() 连在一起使用
        while not it.finished:
            # 获取当前的状态state
            s = it.iterindex
            # 获取当前状态所在grid格子中的值
            y,x = it.multi_index
            # print(y,x)
            # P[s][a] == [(probability,nextstate,reward,done)*4]
            P[s] = {a : [] for a in range(nA)}
            
            s_letter = self.desc[y][x]
            # 使用lambda表达式代替函数,遇到G或者X结束
            is_done = lambda letter : letter in b'GX'
            # 只有到达位置G奖励才为1
            reward = 1.0 if s_letter in b'G'else -1.0
            # reward = -1.0 if s_letter in b'X'else reward


            if is_done(s_letter):
            # 如果达到状态G，直接更新动作-状态转换概率
                P[s][UP] = [1.0,s,reward,True]
                P[s][RIGHT] = [1.0,s,reward,True]
                P[s][DOWN] = [1.0,s,reward,True]
                P[s][LEFT] = [1.0,s,reward,True]
            else:
            # 如果还没有达到状态G
                # 新状态位置的索引
                ns_up = s if y == 0 else s - MAX_X #减去一行的数量，即为上一行的坐标位置
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1

                # 新状态位置的索引对应的字母  // 表示显示浮点数的结果
                sl_up = self.desc[ns_up // MAX_Y][ns_up % MAX_X]
                sl_right = self.desc[ns_right // MAX_Y][ns_right % MAX_X]
                sl_down = self.desc[ns_down // MAX_Y][ns_down % MAX_X]
                sl_left = self.desc[ns_left // MAX_Y][ns_left % MAX_X]

                # 更新动作-状态转换概率
                P[s][UP] = [1.0, ns_up , reward, is_done(sl_up)]
                P[s][RIGHT] = [1.0, ns_right , reward, is_done(sl_right)]
                P[s][DOWN] = [1.0, ns_down , reward, is_done(sl_down)]
                P[s][LEFT] = [1.0, ns_left , reward, is_done(sl_left)]
            # 准备更新下一个状态
            it.iternext()
        self.P = P
        super(HelloGridEnv, self).__init__(nS, nA , P , isd)
    def render(self, mode = 'human', close=False):
        # 判断程序是否结束
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # 格式转化
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line ] for line in desc]

        state_grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(state_grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y,x = it.multi_index

            # 对于当前的状态用红色标志
            if self.s == s:
                desc[y][x] = utils.colorize(desc[y][x],"red",highlight=True)
            it.iternext()
        outfile.write("\n".join(' '.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile
    
    def step(self, a):        
        transitions = self.P[self.s]
        p, s, r, d= transitions[a]
        self.s = s
        # print(self.s)
        self.lastaction = a
        return (int(s), r, d, {"prob" : p})

##################################################################################
# 策略评估：
def policy_evaluation(policy, environment, discount_factor = 1.0, theta = 0.99):
    # 迭代式策略评估
    env = environment

    # 初始化一个全0的值函数向量用于记录状态值
    V = np.zeros(env.nS)
    
    # 迭代开始，10000为最大迭代次数
    for _ in range(10000):
        delta = 0

        # 对于HelloGrid 中的每一个状态都进行全备份
        for s in range(env.nS):
            v = 0
            # 检查下一个有可能执行的动作
            for a, action_prob in enumerate(policy[s]):
                # 对于每一个动作检查下一个状态
                [prob, next_state, reward, done] = env.P[s][a]
                # 累计计算下一个动作价值的期望
                v += action_prob * prob * (reward + discount_factor* V[next_state]) 
            # 选出变化最大的量
            delta = max(delta, np.abs(v-V[s]))
            V[s] = v

        # 检查是否满足停止条件
        if delta <= theta:
            break
    return np.array(V)

# 策略迭代算法
def policy_iteration(env, policy, discount_factor = 1.0):
    while True:
        # 评估当前的策略policy
        # print("policy",np.reshape(np.argmax(policy,axis=1),env.shape))
        V = policy_evaluation(policy,env,discount_factor)
        # print("V",V.reshape(env.shape))
        # policy标志位，当某种的策略更改后，该标志位为False
        policy_stable = True
        # 策略改进
        for s in range(env.nS):
            # 在当前状态和策略下，选择概率最高的动作
            old_action = np.argmax(policy[s])
            # 在当前状态和策略下，找到最优动作
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                [prob, next_state, reward, done] = env.P[s][a]
                action_values[a] += prob*(reward + discount_factor * V[next_state])

                # 由于Grid World环境存在状态遇到陷阱X则停止，因此让状态值遇到陷阱则为负无穷，不参与计算
                if done and next_state != 15:
                    action_values[a] = float('-inf')
            # 采用贪婪算法更新当前策略
            best_action = np.argmax(action_values)
            if old_action != best_action:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_action]
        # 选择的动作不在变化，则代表策略已经稳定下来
        if policy_stable:
            return policy, V
######################################################################
# 值迭代算法
def calc_action_value(state, V, discount_factor = 1.0):
    # 对于给定的状态s计算其动作的期望值

    # 初始化动作值得期望向量[0,0,0,0]
    A = np.zeros(env.nA)

    # 遍历当前状态下得所有动作
    for a in range(env.nA):
        [prob, next_state,reward ,done ] =  env.P[state][a]
        A[a] += prob *(reward + discount_factor * V[next_state])

    return A

def value_iteration(env, theta = 0.1, discount_factor = 1.0):
    # 值迭代算法

    # 初始化状态值
    V = np.zeros(env.nS)

    # 迭代计算找到最优得状态值函数
    for _ in range(50):
        # 停止标志位
        delta = 0

        # 计算每个状态得状态值
        for s in range(env.nS):
            # 执行一次找到当前状态得动作期望
            A = calc_action_value(s,V)
            # 选择最好得动作期望作为新得状态值
            best_action_value = np.max(A)

            # 计算停止得标志位
            delta = max(delta, np.abs(best_action_value - V[s]))

            # 更新状态值函数
            V[s] = best_action_value
        
        if delta < theta:
            break
    
    #输出最优策略：通过最优状态值函数找到确定性策略，并初始化策略
    policy = np.zeros([env.nS,env.nA])

    for s in range(env.nS):
        # 执行一次找到当前状态值得最优状态值得动作期望A
        A = calc_action_value(s,V)
        # 选出最大得状态值作为最优动作
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    return policy,V 

if __name__ == "__main__":
    
    env = HelloGridEnv() # 使用HelloGrid 环境
    state = env.reset() # 初始化
    # 生成随机策略：
    random_policy = np.ones([env.nS,env.nA])/env.nA
    #####################################################
    # # 策略评估验证
    # v = policy_evaluation(random_policy,env)
    # print(v.reshape(env.shape))
    ####################################################
    # 策略迭代算法
    policy,v = policy_iteration(env, random_policy)
    print(np.reshape(np.argmax(policy,axis=1),env.shape))
    print(v.reshape(env.shape))
    ######################################################
    # # 值迭代算法
    # policy, v = value_iteration(env)
    # print(v.reshape(env.shape))
    # print(np.reshape(np.argmax(policy,axis=1),env.shape))

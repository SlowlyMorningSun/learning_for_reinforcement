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
# q_value = []  ###   需要修改
# # 贪婪算法
# def epsilon_greedy( nA , R, T, epsilon = 0.6):
#     # 初始化累计奖励r
#     r = 0
#     N = [0] * nA

#     for _ in range(T):
#         if np.random.rand() < epsilon :
#             # 探索阶段：以均匀分布随机选择
#             a = np.random.randint(q_value.shape[0])
#         else:
#             # 利用阶段：选择价值函数最大的动作
#             a = np.argmax(q_value[:])
        
#         # 更新累计奖励和价值函数
#         v = R(a)
#         r = r+v

#         q_value[a] = (q_value[a] * N[a] + v)/(N[a]+1)
#         N[a] += 1

#         # 返回累计奖励r
#         return r

class HelloGridEnv(discrete.DiscreteEnv):
    metadata = {'render.modes':['human','ansi']}
    
    def __init__(self, desc = None, map_name = '4x4'):
        """ GridWorldEnv 环境构建"""
        # 环境地图Grid
        self.desc = np.asarray(MAPS[map_name],dtype='c')
        # 获取maps的形状（4，4）
        self.shape = self.desc.shape
        # print(self.shape)
        # 动作集的个数
        nA = 4
        # 状态集的个数
        nS = np.prod(self.desc.shape)
        # print(self.desc)
        # 设置最大的行号和最大的列号方便索引
        MAX_Y = self.shape[0]
        MAX_X = self.shape[1]

        # 初始状态分布[1. 0. 0.  ...],并于格子S开始执行
        isd = np.array(self.desc == b'S').astype('float64').ravel()  # 将其变为对应的一维数组
        isd /= isd.sum()
        # print(isd)
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
            reward = 1.0 if s_letter in b'G'else 0.0
            reward = -1.0 if s_letter in b'X'else reward


            if is_done(s_letter):
            # 如果达到状态G，直接更新动作-状态转换概率
                P[s][UP] = [(1.0,s,reward,True)]
                P[s][RIGHT] = [(1.0,s,reward,True)]
                P[s][DOWN] = [(1.0,s,reward,True)]
                P[s][LEFT] = [(1.0,s,reward,True)]
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
        # print(P)
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
        # print(self.P[self.s])
        
        transitions = self.P[self.s]
        # print(transitions)
        # for key,value in transitions.items():
        #     print(value)
        # print(t for t in transitions)
        # i = categorical_sample([key for key,value in transitions.items()], self.np_random.randint(0,10)) # 有问题
        # print(self.np_random.rand())
        p, s, r, d= transitions[a]
        self.s = s
        # print(self.s)
        self.lastaction = a
        return (int(s), r, d, {"prob" : p})

if __name__ == "__main__":
    
    env = HelloGridEnv() # 使用HelloGrid 环境
    state = env.reset() # 初始化

    # 执行5次动作
    for _ in range(5):
        # 显示环境
        env.render()

        # 随机获取动作 action
        action = env.action_space.sample()
        # print(action)
        # print(env.P)
        # 执行随机选取的动作action
        state ,reward, done, info = env.step(action)

        print("action{}({})".format(action ,["Up","Right","Down","Left"][action]))
        print("done:{}, observation:{},reward:{}".format(done,state,reward))

        # 如果执行动作后返回的done状态为True则停止执行
        if done:
            print("Episode finished after {} timesteps".format(_+1))
            break


import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import random
import copy
from gym.envs.classic_control import rendering
from gym import wrappers
import os
import sys
# from pyglet.gl.gl import PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC

#障碍物类，模块的边界，判断液滴是否在障碍内，是否交叉
class Module:
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min > x_max or y_min > y_max:
            raise TypeError('Module() inputs are illegal')
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    def isPointInside(self, point):
        ''' point is in the form of (x, y) '''
        for i in point:
            if i[0] >= self.x_min and i[0] <= self.x_max and\
                    i[1] >= self.y_min and i[1] <= self.y_max:
                return True
        else:
            return False
    def isModuleOverlap(self, m):
        if self._isLinesOverlap(self.x_min, self.x_max, m.x_min, m.x_max) and\
                self._isLinesOverlap(self.y_min, self.y_max, m.y_min, m.y_max):
            return True
        else:
            return False

    def _isLinesOverlap(self, xa_1, xa_2, xb_1, xb_2):
        if xa_1 > xb_2:
            return False
        elif xb_1 > xa_2:
            return False
        else:
            return True

class Dropletenv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2}
    #环境初始化
    def __init__(self,width,length,drop_num,block_num=0):
        super(Dropletenv,self).__init__()
        self.width=width #vertical cell number
        self.length=length # horizon cell number
        self.single_action_space=spaces.Discrete(5) #单个液滴动作空间
        self.action_space=[]
        self.observation_space=[]
        self.reward_range=(-10,10)
        self.past_location=None
        #液滴数量
        self.agent_number=drop_num
        self.agents_current_positon, self.agents_goal=self._GenerateStartEnd()
        self.state = np.vstack((self.agents_current_positon, self.agents_goal))#液滴状态,用于判断。。
        self.n_modules=block_num #障碍数量
        self.modules=self._genRandomModules()
        self.global_state=self._get_global_state() #全局状态
        #最大运行步长
        self.max_step=20
        #绘图用
        self.agent=[None]*self.agent_number
        self.agenttrans=[None]*self.agent_number
        self.step_number = 0  #步长
        self.u_size=40 #size for cell pixels
        self.env_width = self.u_size * self.width    # scenario width (pixels)
        self.env_length = self.u_size * self.length  # height
        self.viewer = None
        # print('success')
        
   # 生成随机障碍区
    def _genRandomModules(self):
        """ Generate reandom modules up to n_modules"""
        if self.width < 5 or self.length < 5:
            return []
        if self.n_modules * 4 / (self.width * self.length) > 0.2:
            print('Too many required modules in the environment.')
            return []
        modules = []
        for i in range(self.n_modules):
            x = random.randrange(0, self.length - 1)
            y = random.randrange(0, self.width - 1)
            m = Module(x, x+1, y, y+1)

            while m.isPointInside(self.state) or \
                    self._isModuleoverlap(m, modules):
                x = random.randrange(0, self.length - 1)
                y = random.randrange(0, self.width - 1)
                m = Module(x, x+1, y, y+1)
            modules.append(m)
        return modules  
    # def _get_MOdules_positons(self):
    #     module_list=[]
    #     for i in self.modules:

    #     pass  
    #初始起点终点生成
    def _GenerateStartEnd(self):
        y=np.random.randint(0,self.width,size=(self.agent_number,1))
        x=np.random.randint(0,self.length,size=(self.agent_number,1))
        Start=np.hstack((x,y))
        y2=np.random.randint(0,self.width,size=(self.agent_number,1))
        x2=np.random.randint(0,self.length,size=(self.agent_number,1))
        Goal=np.hstack((x2,y2))
        while self._initialization_conflict(np.vstack((Start,Goal))):
            y=np.random.randint(0,self.width,size=(self.agent_number,1))
            x=np.random.randint(0,self.length,size=(self.agent_number,1))
            Start=np.hstack((x,y))
            y2=np.random.randint(0,self.width,size=(self.agent_number,1))
            x2=np.random.randint(0,self.length,size=(self.agent_number,1))
            Goal=np.hstack((x2,y2))
        return  Start,Goal
    #判断初始化是否满足静态约束，满足返回False（无冲突），不满足返回True(有冲突)
    def _initialization_conflict(self,points):
        num=points.shape[0]
        for i in range(0,num-1):
            for j in range(i+1,num):
                if np.linalg.norm(points[i]-points[j])<2:
                    return True
        return False
    def _is_comflic_static(self):
        static_conflic=[0]*self.agent_number
        for i in range(self.agent_number-1):
            for j in range(i+1,self.agent_number):
                if np.linalg.norm(self.agents_current_positon[i] - self.agents_current_positon[j]) < 2:
                    static_conflic[i] +=1
                    static_conflic[j] +=1
        return static_conflic
    def _is_conflic_dynamic(self):
        dynamic_conflict = [0] * len(self.agents_current_positon)
        for i in range(self.agent_number):
            for j in range(self.agent_number):
                if i != j:
                    if np.linalg.norm(self.past_location[i] - self.agents_current_positon[j]) < 2:
                        dynamic_conflict[i] +=1
                        dynamic_conflict[j] +=1
        return dynamic_conflict
    #渲染，判断是不是终点    
    def _is_grid_goal(self,cell):
        for i in range(self.agents_goal.shape[0]):
            if np.array_equal(cell,self.agents_goal[i]):
                return i+1
        return False 

    def _isTouchingModule(self, point):
        for m in self.modules:
            if point[0] >= m.x_min and\
                    point[0] <= m.x_max and\
                    point[1] >= m.y_min and\
                    point[1] <= m.y_max:
                return True
        return False
    def _isModuleoverlap(self, m, modules):
        for mdl in modules:
            if mdl.isModuleOverlap(m):
                return True
        return False  
    def _updatePosition(self,action):
    #0 静止 1上 2下 3左 4右
        for i,a in enumerate(action):
            if a==1:
                self.agents_current_positon[i][1]+=1
            elif a==2:
                self.agents_current_positon[i][1]-=1
            elif a==3:
                self.agents_current_positon[i][0]-=1
            elif a==4:
                self.agents_current_positon[i][0]+=1
            if not self._isPointInside(i):
                self.agents_current_positon[i] = self.past_location[i]
            if self._isTouchingModule(self.agents_current_positon[i]):
                self.agents_current_positon[i] = self.past_location[i]
        return None
    def _isPointInside(self,index):
        if self.agents_current_positon[index][0] < 0 or self.agents_current_positon[index][0] > (self.length-1):
            # print('false')
            return False
        elif self.agents_current_positon[index][1] < 0 or self.agents_current_positon[index][1] > (self.width-1):
            # print('false')
            return False
        # print('true')
        return True    
    def step(self,action):
        self.step_number += 1
        self.past_location=copy.deepcopy(self.agents_current_positon)
        self._updatePosition(action)
        self.state = np.vstack((self.agents_current_positon, self.agents_goal))#液滴状态
        obs_n = self._get_observation()
        if self.step_number>self.max_step:
            done_n = [True]*self.agent_number
        else:
            done_n = self._isComplete()
        #静态约束
        sta = np.asarray(self._is_comflic_static())
        #动态约束
        dy = np.asarray(self._is_conflic_dynamic())
        #距离 array
        pre_distance = self._compute_distance(self.past_location)
        lat_distance = self._compute_distance(self.agents_current_positon)
        #奖励值的第一项
        erro=np.asarray([-0.1 if i < 0 else -0.5 for i in lat_distance-pre_distance])
        #done 取反数字化
        reverse_done=np.asarray([0 if i else 1 for i in done_n])
        #单个智能体的奖励值
        reward_n=erro*reverse_done-sta-dy
        #平均奖励值
        ave_reward=[np.sum(reward_n)/len(reward_n)]*self.agent_number
        info_n={}

        #debug
        a=self.global_state
        print('当前状态1\n',a[:,:,0])
        print('当前状态2\n',a[:,:,1])
        print('当前状态3\n',a[:,:,2])
        # print('单个液滴奖励',reward_n)
        # print('平均奖励',ave_reward)
        # print("前一次距离",pre_distance)
        # print('当前距离',lat_distance)
        # print('是否完成',done_n)
        # print('非done', reverse_done)
        # print('静态',sta)
        # print('动态',dy)
        # print(self.agents_current_positon)
        # print('----------------------------------------')
        return obs_n , ave_reward , done_n, info_n

    # def _get_global_state(self):
    #     """
    #     format of state one hot vector + id
    #     l-w-4
    #     Droplet        in layer 0  [1 0 0 id]
    #     Goal           in layer 1  [0 1 0 id]
    #     Obstacles      in layer 2  [0 0 1  0]
    #     """
    #     state = np.zeros((self.length,self.width,4), dtype=np.float32)
    #     state = self._addModulesInObs(state)
    #     # print(state)
    #     # agent_id=0
    #     for i in range(self.agent_number):
    #         #液滴位置处，第一个chanel置1，第四个chanel为液滴编号
    #         state[self.agents_current_positon[i][0]][self.agents_current_positon[i][1]][0]=1
    #         state[self.agents_current_positon[i][0]][self.agents_current_positon[i][1]][3]=i+1
    #         state[self.agents_goal[i][0]][self.agents_goal[i][1]][1]=1
    #         state[self.agents_goal[i][0]][self.agents_goal[i][1]][3]=i+1
    #     return state.reshape(-1)
    #修改于2121.4.20
    def _get_global_state(self):
        """
        format of state one hot vector + id
        l-w-3
        Droplet        in layer 0  [id 0 0]
        Goal           in layer 1  [0 id 0]
        Obstacles      in layer 2  [0 0  1]
        """
        state = np.zeros((self.length,self.width,3), dtype=np.float32)
        state = self._addModulesInObs(state)
        # agent_id=0
        for i in range(self.agent_number):
            state[self.agents_current_positon[i][0]][self.agents_current_positon[i][1]][0]=i+1
            state[self.agents_goal[i][0]][self.agents_goal[i][1]][1]=i+1
        # return state.reshape(-1)
        return state
    def _get_observation(self):
        obs_n=[]
        self.global_state=self._get_global_state()
        # singobs=self.global_state.reshape(-1)
        for i in range(self.agent_number):
            # temp=np.hstack((self.global_state,np.array([i])))
            # obs_n.append(temp)
            #修改2021.4.20
            obs_n.append(self.global_state)
        return obs_n
    def _compute_distance(self,position):
        # (agent_num,)
        return np.sum(np.abs(position - self.agents_goal), axis=1)
    def _isComplete(self):
        done = list(np.all((self.agents_current_positon==self.agents_goal), axis=1))
        return done
    def _addModulesInObs(self, state):
        # if self.n_modules==1:
        #     temp=self.modules
        #     for x in range(temp.x_min, temp.x_max + 1):
        #         for y in range(temp.y_min, temp.y_max + 1):
        #             state[x][y][2] = 1 
        # else:
        for m in self.modules:
            for x in range(m.x_min, m.x_max + 1):
                for y in range(m.y_min, m.y_max + 1):
                    state[x][y][2] = 1 
        return state

    def reset(self):
        self.agents_current_positon,self.agents_goal=self._GenerateStartEnd()
        self.state = np.vstack((self.agents_current_positon, self.agents_goal))#液滴状态
        self.modules=self._genRandomModules()
        self.global_state=self._get_global_state() #全局状态
        # obs=np.hstack((self.agents_current_positon, self.agents_goal))
        obs=self._get_observation()
        self.step_number=0
        # self.viewer = None
        # self.state= obs
        return obs
    def render(self,mode='human',close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return None
        u_size = 40
        m=2
        if self.viewer is None :
            self.viewer = rendering.Viewer(self.env_length , self.env_width)
        for x in range(self.length):
            for y in range(self.width):
                v=[(x*u_size+m,y*u_size+m),
                ((x+1)*u_size-m,y*u_size+m),
                ((x+1)*u_size-m,(y+1)*u_size-m),
                (x*u_size+m,(y+1)*u_size-m)]
                rect = rendering.FilledPolygon(v)
                temp=np.array([x,y])
                a=self._is_grid_goal(temp)
                if a:
                #终点
                    rect.set_color(0.1, 0.6, 1.0/self.agent_number*a)
                elif self._isTouchingModule(temp):
                #block
                    rect.set_color(0.0,0.0,0.0)
                else:
                #普通格子
                    rect.set_color(0.9,0.9,0.9)
                self.viewer.add_geom(rect)
        for i in range(self.agent_number):
            self.agent[i] = rendering.make_circle(u_size/2.5, 50, True)
            self.agent[i].set_color(0.1, 0.6+0.05, 1.0/self.agent_number*(i+1))
            self.viewer.add_geom(self.agent[i])
            self.agenttrans[i]=rendering.Transform()
            self.agent[i].add_attr(self.agenttrans[i])
        for i in range(self.agent_number):
            [x,y]=self.agents_current_positon[i]
            self.agenttrans[i].set_translation((x+0.5)*u_size, (y+0.5)*u_size)    
        return self.viewer.render( return_rgb_array = mode=='rgb_array')
    
    def close(self):
        self.viewer.close()

if __name__ == '__main__':
    env = Dropletenv()
    # env=wrappers.Monitor(env,'./experiment-1')
    # print(env.state)
    agent_num=env.agent_number
    for i in range(10):
        env.reset()
        # action=[1]*env.agent_number
        for j in range(10):
            action=[]
            for k in range(agent_num):
                action.append(env.single_action_space.sample())
        # print(action)
            obs_n , reward_n,done_n,info_n=env.step(action)
            env.render()
            time.sleep(1)
            # print(env.state)
            # print('reward',reward_n)
            # done_n=np.array(done_n)
            # flag=np.all(done_n)
            # if flag:
            #     print("has break")
            #     break
        # print(obs_n)
    env.close()
        
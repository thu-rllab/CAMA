import numpy as np
import cv2
from copy import deepcopy
from ..multiagentenv import MultiAgentEnv


class CatchApple(MultiAgentEnv):
    def __init__(self, entity_scheme=True, n_agents=2, n_wall=2, episode_limit=50, size_x = 9, size_y=9, sr=3, \
                fence_len=3, coord_reward = 10, single_reward =1, full_observe=False, all_agents_observe=False, 
                raw_state=True, transparent_wall=False, seed=None):
        super(CatchApple, self).__init__()
        self.n_agents = n_agents
        self.n_wall = n_wall
        self.entity_scheme = entity_scheme
        self.episode_limit = episode_limit
        self.size_x = size_x
        self.size_y = size_y
        self.sr = sr
        assert sr % 2 == 1, "Only supprt odd sr!"
        self.coord_reward = coord_reward
        self.single_reward = single_reward
        self.fence_len = fence_len
        self.full_observe = full_observe
        self.transparent_wall = transparent_wall
        self.raw_state = raw_state
        self.all_agents_observe = all_agents_observe
        self.n_actions = 5 #four directions and stay
        self.entity_size = self.n_agents + 1 + self.fence_len * self.n_wall
        self.move_vec = [[0,0],[0,1],[0,-1],[-1,0],[1,0]]
        self.seed(seed)
    
    
    def step(self, actions):
        actions = [int(a) for a in actions[:self.n_agents]]
        apple_caught= 0
        done = False
        reward = 0
        for i in range(self.n_agents):
            if actions[i]> 0:
                v_pos = [self.agent_pos[i][0] + self.move_vec[actions[i]][0], self.agent_pos[i][1] + self.move_vec[actions[i]][1]]
                v_pos[0] = np.clip(v_pos[0], 0, self.size_x - 1)
                v_pos[1] = np.clip(v_pos[1], 0, self.size_y - 1)
                tp_k = self.maze[v_pos[0], v_pos[1]]
                if tp_k == 0:
                    self.maze[v_pos[0], v_pos[1]]=3
                    self.maze[self.agent_pos[i][0], self.agent_pos[i][1]]=0
                    self.agent_pos[i] = [v_pos[0], v_pos[1]]
                elif tp_k == 2:
                    apple_caught += 1
                    self.agent_pos[i] = [v_pos[0], v_pos[1]]
        info = {'solved': False}
        if apple_caught >= 1:
            done=True
            if apple_caught == self.n_wall:
                reward += self.coord_reward
                info['solved'] = True
            else:
                reward += self.single_reward
        else:
            reward -= 0.1
        self.t += 1
        if self.t == self.episode_limit:
            done = True
            info['episode_limit'] = True
        return reward, done, info
            
    def can_observe(self, pos_i, pos_j):
        thre = self.sr // 2
        if abs(pos_i[0]-pos_j[0]) <= thre and abs(pos_i[1]-pos_j[1]) <= thre:
            return True
        else:
            return False
    
    def direct_observe(self, pos_i, pos_j): #only work for sr == 3 or 5
        if self.transparent_wall:
            return True
        checklist = []
        if abs(pos_j[0]-pos_i[0]) >= abs(pos_j[1]-pos_i[1]): #x>y, for on x
            if abs(pos_j[0]-pos_i[0]) <= 1: #neighbor, can see.
                return True
            x = (pos_i[0] + pos_j[0]) // 2 
            y = pos_i[1]+(x-pos_i[0])*(pos_j[1]-pos_i[1])/(pos_j[0]-pos_i[0])
            if y-np.floor(y) == 0.5:
                checklist.append([x, int(np.floor(y))])
                checklist.append([x, int(np.ceil(y))])
            else:
                checklist.append([x, int(np.rint(y))])
        else: #y>x, for on y
            if abs(pos_j[1]-pos_i[1]) <= 1:
                return True
            y = (pos_i[1]+pos_j[1]) // 2
            x = pos_i[0]+(y-pos_i[1])*(pos_j[0]-pos_i[0])/(pos_j[1]-pos_i[1])
            if x - np.floor(x) == 0.5:
                checklist.append([int(np.floor(x)), y])
                checklist.append([int(np.ceil(x)), y])
            else:
                checklist.append([int(np.rint(x)), y])
        for i in checklist:
            if self.maze[i[0],i[1]] > 0:
                return False
        return True

    def get_masks(self):
        obs_mask = np.zeros([self.entity_size,self.entity_size])
        entity_mask = np.zeros(self.entity_size)
        if self.full_observe:
            return obs_mask, entity_mask
        for i in range(self.n_agents):
            obs_mask[i,:] = 1
            for j in range(self.n_agents):
                if self.can_observe(self.agent_pos[i], self.agent_pos[j]) and self.direct_observe(self.agent_pos[i], self.agent_pos[j]):
                    obs_mask[i,j] = 0
            ind = self.n_agents
            for j in range(len(self.fence_pos)):
                if self.can_observe(self.agent_pos[i], self.fence_pos[j]) and self.direct_observe(self.agent_pos[i], self.fence_pos[j]):
                    obs_mask[i, ind+j] = 0
            ind += len(self.fence_pos)
            if self.can_observe(self.agent_pos[i], [self.apple_x, self.apple_y]) and self.direct_observe(self.agent_pos[i], [self.apple_x, self.apple_y]):
                obs_mask[i, ind] = 0
        if self.all_agents_observe:
            for i in range(1, self.n_agents):
                obs_mask[0,:] = 1-np.clip((1-obs_mask[0,:])+(1-obs_mask[i,:]), 0, 1)
            for i in range(1, self.n_agents):
                obs_mask[i,:] = obs_mask[0, :]
            
        return obs_mask, entity_mask

    def get_entities(self):
        entities = []
        for i in range(self.n_agents):
            entities.append(np.hstack([np.array(self.pos2obs(self.agent_pos[i])), np.array([1,0,0])]))
        for i in range(len(self.fence_pos)):
            entities.append(np.hstack([np.array(self.pos2obs(self.fence_pos[i])), np.array([0,1,0])]))
        entities.append(np.hstack([np.array(self.pos2obs([self.apple_x, self.apple_y])), np.array([0,0,1])]))
        return entities
        
    def pos2obs(self, pos):
        return [pos[0]/(self.size_x-1), pos[1]/(self.size_y-1)]

    def get_avail_actions(self):
        return [[1 for _ in range(self.n_actions)] for _ in range(self.n_agents)]

    def reset(self, **kwargs):
        #0:available; 1:fence; 2:apple; 3:agent
        self.maze = np.zeros([self.size_x, self.size_y], dtype=int)
        self.apple_x = self.random.randint(self.fence_len, self.size_x-self.fence_len)
        self.apple_y = self.random.randint(self.fence_len, self.size_y-self.fence_len)
        self.maze[self.apple_x, self.apple_y] = 2
        self.fence_pos = []
        if self.n_wall == 2:
            if self.random.uniform()>0.5:
                for x in range(self.apple_x-self.fence_len, self.apple_x+self.fence_len+1):
                    if not x == self.apple_x:
                        self.maze[x, self.apple_y] = 1
                        self.fence_pos.append([x, self.apple_y])
            else:
                for y in range(self.apple_y-self.fence_len, self.apple_y+self.fence_len+1):
                    if not y == self.apple_y:
                        self.maze[self.apple_x, y] = 1
                        self.fence_pos.append([self.apple_x, y])
        elif self.n_wall == 3:
            d = int(np.floor(self.random.uniform()*4))
            f_d =[[[1,0],[-1,-1],[-1,1]],
                  [[0,-1],[-1,1],[1,1]],
                  [[-1,0],[1,1],[1,-1]],
                  [[0,1],[1,-1],[-1,-1]]]
            d=f_d[d]
            for i in range(3):
                sx, sy = self.apple_x, self.apple_y
                for _ in range(self.fence_len-1): #in case the fence block the way
                    sx += d[i][0]
                    sy += d[i][1]
                    self.maze[sx, sy] = 1
                    self.fence_pos.append([sx, sy])
        else:
            raise NotImplementedError
        self.agent_pos = []
        for _ in range(self.n_agents):
            x = self.random.randint(self.size_x)
            y = self.random.randint(self.size_y)
            while self.maze[x,y] > 0:
                x = self.random.randint(self.size_x)
                y = self.random.randint(self.size_y)
            self.maze[x,y]=3
            self.agent_pos.append([x,y])
        self.t = 0
        self.not_rendered = True
        self.saved_img = None
        if self.entity_scheme:
            return self.get_entities(), self.get_masks()
        else:
            return self.get_obs(), self.get_state()

    def close(self):
        return
    
    def get_center(self, pos, img_size):
        x = int((2*pos[0] + 1) * (img_size[0] / self.size_x / 2))
        y = int((2*pos[1] + 1) * (img_size[1] / self.size_y / 2))
        return [x, y]

    def render(self):
        imgX = 512
        imgY = 512
        widthX = int(np.floor((imgX/self.size_x - 5) / 2))
        widthY = int(np.floor((imgY/self.size_y - 5) / 2))
        if self.not_rendered:
            img = np.ones((imgX, imgY,3), np.uint8)*255
            for i in range(self.size_x+1):
                cx = int(imgX / self.size_x * i)
                cv2.line(img,(cx,0),(cx, imgY),(0,0,0),5)
            for i in range(self.size_y+1):
                cy = int(imgY / self.size_y * i)
                cv2.line(img,(0,cy),(imgX, cy),(0,0,0),5)
            cv2.ellipse(img,self.get_center([self.apple_x,self.apple_y], [imgX, imgY]),(widthX,widthY),0,0,360,(0, 0, 255),-1)
            for f in self.fence_pos:
                cpos = self.get_center(f, [imgX, imgY])
                cv2.rectangle(img,(cpos[0]-widthX, cpos[1]-widthY),(cpos[0] + widthX, cpos[1]+widthY),(75,108,130),-1)
            self.saved_img = deepcopy(img)
            self.not_rendered = False
        else:
            img = deepcopy(self.saved_img)
        for a in self.agent_pos:
            apos = self.get_center(a, [imgX, imgY])
            dp = 6
            cv2.rectangle(img,(apos[0]-widthX+dp, apos[1]-widthY+dp),(apos[0] + widthX-dp, apos[1]+widthY-dp),(255,0,0),-1)
        return img[:,:,(2,1,0)] #BGR to RGB
    
    def get_entity_shape(self):
        return 5
    
    def seed(self, seed):
        if seed is None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(seed)
        
    def get_obs_shape(self):
        if self.raw_state:
            return self.sr**2
        else:
            return self.get_state_shape()
    
    def get_state_shape(self):
        if self.raw_state:
            return self.size_x * self.size_y
        else:
            return self.entity_size * self.get_entity_shape()

    def get_obs(self):
        if self.raw_state:
            obs = []
            for i in range(self.n_agents):
                single_obs = np.zeros([self.sr, self.sr])
                center = self.agent_pos[i]
                d = self.sr // 2
                for x, xx in zip(range(center[0]-d,center[0]+d+1), range(self.sr)):
                    for y, yy in zip(range(center[1]-d, center[1]+d+1), range(self.sr)):
                        if x>=0 and x < self.size_x and y>=0 and y < self.size_y:
                            single_obs[xx,yy] = self.maze[x,y] / 3
                obs.append(single_obs.flatten())
            return obs
        else:
            obs = []
            mask, _ = self.get_masks()
            entities = self.get_entities()
            for i in range(self.n_agents):
                co = np.array(entities)
                obs.append(((1-mask[i].reshape(self.entity_size,1))*co).flatten())
            return obs

    def get_state(self):
        if self.raw_state:
            return self.maze.flatten()/3
        else:
            return np.array(self.get_entities()).flatten()

    def get_env_info(self, args):
        env_info = {"entity_shape": self.get_entity_shape(),
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "n_entities": self.entity_size,
            "episode_limit": self.episode_limit,
            "state_shape":self.get_state_shape(),
            "obs_shape":self.get_obs_shape(),}
        return env_info
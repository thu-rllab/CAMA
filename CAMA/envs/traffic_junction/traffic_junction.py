import numpy as np
from ..multiagentenv import MultiAgentEnv
import cv2
from copy import deepcopy
class Entity_Traffic_Junction_Env(MultiAgentEnv):
    def __init__(self,entity_scheme=True,difficulty='easy',vision=0,seed=0) -> None:
        self.difficulty = difficulty
        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10
        self.TARGET_CLOSE_WEIGHT = 1 #每一步1或-1

        self.vision = vision
        self.curr_end = 625000
        self.curr_start = 125000

        if difficulty == 'easy':
            self.add_rate_max = 0.3
            self.add_rate_min = 0.1
            self.dim = 7
            self.max_steps = 20
            self.road_width = 1
            self.in_out_entrances = 4
            self.ncar = 5
        if difficulty == 'medium':
            self.add_rate_max = 0.2
            self.add_rate_min = 0.05
            self.dim = 14
            self.max_steps = 40
            self.road_width = 2
            self.in_out_entrances = 8
            self.ncar = 10
        if difficulty == 'hard':
            self.add_rate_max = 0.05
            self.add_rate_min = 0.02
            self.curr_end = 6250000
            self.cur_start = 1250000
            # self.add_rate_max = 2
            # self.add_rate_min = 1
            self.dim = 18
            self.max_steps = 80
            self.road_width = 2
            self.in_out_entrances = 8
            self.ncar = 20
        self.exact_rate = self.add_rate = self.add_rate_min
        self.n_actions = 5
        self.entity_size = self.ncar
        self.move_vec = [[0,0],[0,1],[0,-1],[-1,0],[1,0]]
        self._make_grid_world()
        self.epoch_last_update = -1
        self.seed(seed)

    def reset(self, **kwargs):
        self.not_rendered = True
        epoch = kwargs['t_env'] if 't_env' in kwargs.keys() else None
        # set add rate according to the curriculum
        epoch_range = (self.curr_end - self.curr_start)
        add_rate_range = (self.add_rate_max - self.add_rate_min)
        if epoch is not None and epoch_range > 0 and add_rate_range > 0 and epoch > self.epoch_last_update:
            self.curriculum(epoch)
            self.epoch_last_update = epoch

        self.collision_times = 0

        self.entity_mask = np.ones(self.ncar) #if alive
        self.t = 0
        self.wait = np.zeros(self.ncar)
        self.cars_pos = np.zeros((self.ncar,2),dtype=float)
        self.last_cars_pos = np.zeros((self.ncar,2),dtype=float)
        self.cars_target = np.zeros((self.ncar,2),dtype=float)
        self._force_add_car()
        self.time_penalty = 0
        self.target_close_reward = 0
        self.exist_car_num_list = [self.exist_car_num()]

        return self.get_entities(), self.get_masks()

    def step(self,action):
        self.wait+=1
        self.t+=1
        self.last_cars_pos = np.zeros((self.ncar,2),dtype=float)+self.cars_pos
        for ii in range(self.ncar):
            if self.entity_mask[ii] ==0:
                new_pos = self.cars_pos[ii]+self.move_vec[action[ii]]
                if np.min(new_pos)<0 or np.max(new_pos)>self.dim-1:
                    continue
                if self.grid_world[int(new_pos[0]),int(new_pos[1])] == 1:
                    self.cars_pos[ii] = self.cars_pos[ii]+self.move_vec[action[ii]]

        reward = self.get_reward().sum()
        done = self.t>=self.max_steps
        self._add_car()
        self.exist_car_num_list.append(self.exist_car_num())
        info = {}
        if done:
            info = {"collision_times":self.collision_times,
            "time_penalty":self.time_penalty,
            "target_close_reward":self.target_close_reward,
            "exist_car_num": sum(self.exist_car_num_list)/len(self.exist_car_num_list)}
        return reward,done, info
    
    def get_reward(self):
        reward = np.full(self.ncar,self.TIMESTEP_PENALTY)*self.wait*(1-self.entity_mask)
        self.time_penalty += reward.sum()
        for ii in range(self.ncar):
            if self.entity_mask[ii] == 0:
                ###reward for target
                reward[ii]+=(np.sum(np.abs(self.last_cars_pos[ii]-self.cars_target[ii]))-np.sum(np.abs(self.cars_pos[ii]-self.cars_target[ii])))*self.TARGET_CLOSE_WEIGHT
                self.target_close_reward += (np.sum(np.abs(self.last_cars_pos[ii]-self.cars_target[ii]))-np.sum(np.abs(self.cars_pos[ii]-self.cars_target[ii])))*self.TARGET_CLOSE_WEIGHT

                if (self.cars_pos[ii] == self.cars_target[ii]).all():
                    self._remove_car(ii)

        for ii in range(self.ncar):
                #check collision  after collision,the cars will still be alive TODO
            collision_happen = 0
            for jj in range(self.ncar):
                if ii!=jj:
                    if self.entity_mask[jj] == 0 and (self.cars_pos[ii] == self.cars_pos[jj]).all():
                        
                        self.collision_times += 1
                        collision_happen +=1
                        self._remove_car(jj)
            
            if collision_happen>0:
                reward[ii]+=self.CRASH_PENALTY
                self._remove_car(ii)
                            
                            # print(self.get_entities())
        if self.difficulty == 'hard':
            reward /= 5
        return reward

    def _remove_car(self,id):
        self.wait[id]=0
        self.cars_pos[id]*=0
        self.last_cars_pos[id]*=0
        self.entity_mask[id] = 1
        self.cars_target[id]*=0

    def _force_add_car(self):
        ii = np.random.choice(range(self.ncar))
        idx = np.random.choice(np.arange(self.in_out_entrances))
        arrival_pos = self.in_entrances_list[idx]
        while True:
            target_idx = np.random.choice(np.arange(self.in_out_entrances))
            target_pos = self.out_entrances_list[target_idx]
            if np.max(np.abs(np.array(arrival_pos)-np.array(target_pos)))>1:
                break
        self.entity_mask[ii] = 0
        self.cars_pos[ii] = arrival_pos
        self.cars_target[ii] = target_pos
    
    def exist_car_num(self):
        return self.ncar - np.sum(self.entity_mask)

    def _add_car(self):
        if np.sum(self.entity_mask) == 0 :
                return
        elif np.sum(self.entity_mask) == self.ncar :
            self._force_add_car()
        else:
            for ii in range(self.ncar):
                if self.entity_mask[ii] and np.random.uniform() <= self.add_rate:
                    can_add = False
                    for _ in range(self.in_out_entrances):
                        idx = np.random.choice(np.arange(self.in_out_entrances))
                        arrival_pos = self.in_entrances_list[idx]
                        if_collision = 0
                        for jj in range(self.ncar):
                            if (self.cars_pos[jj] == arrival_pos).all() and ii!=jj:
                                if_collision=1
                        if not if_collision:
                            while True:
                                idx = np.random.choice(np.arange(self.in_out_entrances))
                                target_pos = self.out_entrances_list[idx]
                                if np.max(np.abs(np.array(arrival_pos)-np.array(target_pos)))>1:
                                    break
                            can_add = True
                            break
                    if can_add:
                        self.entity_mask[ii] = 0
                        self.cars_pos[ii] = arrival_pos
                        self.cars_target[ii] = target_pos

    def get_entities(self):
        entities_list = []
        for ii in range(self.ncar):
            entities_list.append(np.concatenate([self.cars_target[ii],self.cars_pos[ii]])/self.dim)
        return entities_list

    def get_entity_shape(self):
        return 4
    def get_private_entity_shape(self):
        return 2

    def get_masks(self):
        obs_masks = np.ones((self.ncar,self.ncar))
        for ii in range(self.ncar):
            dis = np.max(np.abs(self.cars_pos[ii]-self.cars_pos),axis=-1)#5,
            obs_mask = [1 if dis[jj]>self.vision else 0 for jj in range(self.ncar)]
            obs_masks[ii] = np.zeros(self.ncar)+obs_mask

        return obs_masks, self.entity_mask

    def close(self):
        return

    def get_avail_actions(self):
        return [[1 for _ in range(self.n_actions)] for ii in range(self.ncar)]
            


    def _make_grid_world(self):
        self.grid_world = np.zeros((self.dim,self.dim))
        self.in_entrances_list = []
        self.out_entrances_list = []
        self.in_out_entrances_list = []
        if self.difficulty!='hard':
            self.grid_world[self.dim//2] =1
            self.grid_world[:,self.dim//2]=1
            self.in_entrances_list += self._get_corresponding_entrances(self.dim//2)
            self.out_entrances_list += self._get_corresponding_entrances(self.dim//2)
            self.in_out_entrances_list += self._get_corresponding_entrances(self.dim//2)
            if self.road_width>1:
                self.grid_world[self.dim//2-1] =1
                self.grid_world[:,self.dim//2-1]=1
                self.in_entrances_list += self._get_corresponding_entrances(self.dim//2-1)
                self.out_entrances_list += self._get_corresponding_entrances(self.dim//2-1)
                self.in_out_entrances_list += self._get_corresponding_entrances(self.dim//2-1)
        else:
            self.grid_world[self.dim//3-2] =1
            self.grid_world[:,self.dim//3-2]=1
            self.grid_world[self.dim//3-1] =1
            self.grid_world[:,self.dim//3-1]=1
            self.in_out_entrances_list += self._get_corresponding_entrances(self.dim//3-2)#4
            self.in_out_entrances_list += self._get_corresponding_entrances(self.dim//3-1)#5
            self.grid_world[self.dim//3*2] =1
            self.grid_world[:,self.dim//3*2]=1
            self.grid_world[self.dim//3*2+1] =1
            self.grid_world[:,self.dim//3*2+1]=1
            self.in_out_entrances_list += self._get_corresponding_entrances(self.dim//3*2)
            self.in_out_entrances_list += self._get_corresponding_entrances(self.dim//3*2+1)

            self.in_entrances_list += [[0,4],[0,12],[4,17],[5,0],[12,17],[13,0],[17,5],[17,13]]
            self.out_entrances_list += [[4,0],[12,0],[17,4],[0,5],[17,12],[0,13],[5,17],[13,17]]

    def _get_corresponding_entrances(self,x):
        entries = []
        entries.append([x,0])
        entries.append([x,self.dim-1])
        entries.append([0,x])
        entries.append([self.dim-1,x])
        return entries
    
    def get_env_info(self, args):
        env_info = {"entity_shape": self.get_entity_shape(),
            "private_entity_shape": self.get_private_entity_shape(),
            "n_actions": self.n_actions,
            "n_agents": self.ncar,
            "n_entities": self.entity_size,
            "episode_limit": self.max_steps,}
        return env_info


    
    def _choose_dead(self):
        # all idx
        car_idx = np.arange(len(self.alive_mask))
        # random choice of idx from dead ones.
        return np.random.choice(car_idx[self.alive_mask == 0])
    
    def curriculum(self, epoch):
        step_size = 0.01
        step = (self.add_rate_max - self.add_rate_min) / (self.curr_end - self.curr_start)

        if self.curr_start <= epoch < self.curr_end:
            self.exact_rate = self.exact_rate + step
            self.add_rate = step_size * (self.exact_rate // step_size)
    
    def seed(self, seed):
        if seed is None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(seed)
    
    def render(self):
        widthX = 20#int(np.floor((imgX/self.dim - 5) / 2))
        widthY = 20#int(np.floor((imgY/self.dim - 5) / 2))
        imgX = (widthX*2+5)*self.dim#512
        imgY = (widthY*2+5)*self.dim#512
        if self.not_rendered:
            img = np.ones((imgX, imgY,3), np.uint8)*255
            for i in range(self.dim+1):
                cx = int(imgX / self.dim * i)
                cv2.line(img,(cx,0),(cx, imgY),(0,0,0),widthX//6)
            for i in range(self.dim+1):
                cy = int(imgY / self.dim * i)
                cv2.line(img,(0,cy),(imgX, cy),(0,0,0),widthX//6)
            # cv2.ellipse(img,self.get_center([self.apple_x,self.apple_y], [imgX, imgY]),(widthX,widthY),0,0,360,(0, 0, 255),-1)
            self.saved_img = deepcopy(img)
            self.not_rendered = False
        else:
            img = deepcopy(self.saved_img)
        
        for ii in range(self.dim):
            for jj in range(self.dim):
                if self.grid_world[ii,jj] == 1:
                    apos = self.get_center([ii,jj], [imgX, imgY])
                    dp = widthX//5
                    cv2.rectangle(img,(apos[0]-widthX+dp, apos[1]-widthY+dp),(apos[0] + widthX-dp, apos[1]+widthY-dp),(0,0,0),-1)
        for idx in range(self.ncar):
            if self.entity_mask[idx] == 0:
                apos = self.get_center(self.cars_pos[idx], [imgX, imgY])
                dp = widthX//5*2
                cv2.rectangle(img,(apos[0]-widthX+dp, apos[1]-widthY+dp),(apos[0] + widthX-dp, apos[1]+widthY-dp),(255,0,0),-1)
        return img[:,:,(2,1,0)] #BGR to RGB
    def get_center(self, pos, img_size):
        x = int((2*pos[0] + 1) * (img_size[0] / self.dim / 2))
        y = int((2*pos[1] + 1) * (img_size[1] / self.dim / 2))
        # return [x, y]
        return [y,x]

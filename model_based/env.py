import sys
import json
import torch
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
from DRL.ddpg import decode
from utils.util import *
from PIL import Image
from torchvision import transforms, utils

width = 128

train_num = 0
test_num = 0

class Paint:
    def __init__(self, batch_size, max_step):
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = (13)
        self.observation_space = (self.batch_size, width, width, 7)
        self.test = False
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        global train_num, test_num
        import os
        
        possible_paths = [
            './data/img_align_celeba/',
            '../data/img_align_celeba/',
            '../../data/img_align_celeba/'
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            print(f"Error: Could not find CelebA data at any of {possible_paths}")
            self.train_data = torch.zeros((10, 3, width, width), dtype=torch.uint8)
            self.test_data = torch.zeros((10, 3, width, width), dtype=torch.uint8)
            train_num = 10
            test_num = 10
            print('Device:', device)
            return

        print(f'Loading data from {data_path}...')
        temp_train = []
        temp_test = []
        
        max_images = 40000 
        
        for i in range(max_images):
            img_id = '%06d' % (i + 1)
            file_path = os.path.join(data_path, img_id + '.jpg')
            
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = cv2.resize(img, (width, width))
                if i >= 2000:                
                    temp_train.append(img)
                else:
                    temp_test.append(img)
            
            if (i + 1) % 2000 == 0:
                print('Checked {} images, found {}...'.format(i + 1, len(temp_train) + len(temp_test)), flush=True)
        
        train_num = len(temp_train)
        test_num = len(temp_test)
        
        print('stacking data...', flush=True)
        if train_num > 0:
            self.train_data = torch.from_numpy(np.stack(temp_train)).permute(0, 3, 1, 2)
        if test_num > 0:
            self.test_data = torch.from_numpy(np.stack(temp_test)).permute(0, 3, 1, 2)
            
        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)), flush=True)
        
    def reset(self, test=False, begin_num=0):
        self.test = test
        if test:
            ids = (np.arange(self.batch_size) + begin_num) % test_num
            self.gt = self.test_data[ids].to(device)
        else:
            ids = np.random.randint(train_num, size=self.batch_size)
            self.gt = self.train_data[ids].to(device)
            
            flip_mask = torch.rand(self.batch_size) > 0.5
            self.gt[flip_mask] = torch.flip(self.gt[flip_mask], [3])
            
        self.imgid = ids.tolist()
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0
        self.canvas = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()
    
    def observation(self):
        T = torch.full([self.batch_size, 1, width, width], self.stepnum, dtype=torch.uint8, device=device)
        return torch.cat((self.canvas, self.gt, T), 1) # canvas, img, T

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)
    
    def step(self, action):
        self.canvas = (decode(action, self.canvas.float() / 255) * 255).byte()
        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward()
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)
    
    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)

#!/usr/bin/python
# -*- coding:utf-8 -*-
#coding=utf-8
import numpy as np
import random
from math import *
from item import * 
import tensorflow as tf
from sklearn.decomposition import PCA

def scale2limit(x, xmin, xmax, ymin, ymax):
    ans = (x - xmin)/(xmax - xmin) * (ymax - ymin) + ymin
    ans = ans if ans > ymin else ymin
    ans = ans if ans < ymax else ymax
    return ans
class Action:
    pass

class Observation:
    pass

class Env(object):
    def __init__(self, param):
        self.param = param
        self.action_space = Action()
        self.observation_space = Observation()
        self.action_space.shape = (int(self.param['dim1_action']),)
        self.action_space.low = -1.0
        self.action_space.high = 1.0
        if int(self.param['valid_2d_obs']) == 1:
            self.observation_space.shape = (int(self.param['dim1_state'])//int(self.param['dim1_item_state']),int(self.param['dim1_item_state']),)
        else:
            self.observation_space.shape = (int(self.param['dim1_state']),)
        self.im = ItemModel(self.param)

    def __del__(self):
        self.close()

    def close(self):
        if int(self.param['record']) == 1:
            self.fp_record_item.close()
            self.fp_record_metrics.close()
            self.fp_record_step.close()
            self.fp_record_action.close()
            self.fp_record_state.close()
        
    def reset(self):
        self.item_list = []
        if int(self.param['record']) == 1 :
            self.fp_record_item = open(self.param['filepath_record_item'], 'w')
            self.fp_record_metrics = open(self.param['filepath_record_metrics'], 'w')
            self.fp_record_step = open(self.param['filepath_record_step'], 'w')
            self.fp_record_action = open(self.param['filepath_record_action'], 'w')
            self.fp_record_state = open(self.param['filepath_record_state'], 'w')
        
        if int(self.param['init_from_file']) == 0:
            self.init_random_env()
        else:
            self.load_env(self.param['filepath_env_saved'])
             
        return self.observe(self.t)

    def get_reward(self, t):
        '''get the current reward in the platform'''
        #r=self.reward_01(t)
        r=self.reward_01_normalized(t)
        #r=self.reward_02(t)
        #r=self.reward_03(t)
        return r

    def reward_01_normalized(self, t):
        r = self.reward_01(t)
        # r = (r - 9e3)/1e3
        return r
    
    def reward_01(self, t):
        s1 = np.zeros(4)
        s2 = np.zeros(4)
        c1 = np.zeros(4)
        clk_cnt = 0
        clk_sum = 0
        for item in self.item_list:
            if item.x['stage'] >= 0 and item.x['stage'] <=3:
                s1[item.x['stage']] += item.x['q']
                s2[item.x['stage']] += item.x['dq']
                c1[item.x['stage']] += 1
                clk_cnt += 1
                clk_sum += item.x['click'] / item.x['dt'] if item.x['dt'] >0 else 0.0
        avg1 = s2[1] / (c1[1] if c1[1] > 0 else 1.0)
        avg_maturity = s2[2] / (c1[2] if c1[2] > 0 else 1.0)
        percent_maturity = s2[2] / (s2[0] + s2[1] + s2[2] + s2[3])
        r1 = avg1 # 成长期新品的平均流量作为增量
        s2 =  0.0
        for item in self.item_list:
            s2 += item.x['dq'] * item.x['ctr'] #todo:应该使用动态ctr来计算点击
             
        r2_global_click = s2 / (len(self.item_list) if len(self.item_list) >0 else 1.0)
        #print 'reward r1, r2: ', r1, r2
        #r3 = avg_maturity #
        r3 = percent_maturity
        r4 = clk_sum / clk_cnt 
        r = (r1 / float(self.param['reward_r1_normalize_coef']) * float(self.param['reward_r1_weight']) +
             r2_global_click / float(self.param['reward_r2_normalize_coef']) *  float(self.param['reward_r2_weight']) +
             r3 / float(self.param['reward_r3_normalize_coef']) *  float(self.param['reward_r3_weight']) + 
             r4 / float(self.param['reward_r4_normalize_coef']) *  float(self.param['reward_r4_weight']))/\
            (float(self.param['reward_r1_weight']) + float(self.param['reward_r2_weight']) + float(self.param['reward_r3_weight']) + float(self.param['reward_r4_weight']))
        #r = (r1 / float(self.param['reward_r1_normalize_coef']) * float(self.param['reward_r1_weight']) +
        #      r2 / float(self.param['reward_r2_normalize_coef']) *  float(self.param['reward_r2_weight']) +
        #      r3 / float(self.param['reward_r2_normalize_coef']) *  float(self.param['reward_r3_weight']))/3.0
        return r 
    def reward_02(self, t):
        # 当前时刻进入成熟期的商品的数量 
        new_maturity_cnt = 0
        sum_startup_time = 0
        for item in self.item_list:
            if item.x['stage'] == 1 and (t - item.x['lsct']) > 0 and (t - item.x['lsct'] < float(self.param['new_maturity_fresh_time'])):
                new_maturity_cnt += 1
                sum_startup_time += t - item.x['t0']
        r0 = new_maturity_cnt
        r1 = sum_startup_time / new_maturity_cnt if new_maturity_cnt >0 else 0.0
        return -r1

    def reward_03(self, t):
        sum_dq = 0
        for item in self.item_list:
            if item.x['stage'] == 0:
                sum_dq += item.x['dq']
        r0 = sum_dq
        return r0
    def item_birth_grow(self, t):
        """
        控制新品的生成，当前逻辑是上一轮少了几个宝贝当前轮则生成几个宝贝
        :param t: 
        :return: 
        """
        'give birth to new items and drop dead items'
        self.item_new_count = 0
        n_item_birthrate = float(self.param['n_item_birthrate'])
#        n_item_birth_n = int(self.param['n_item_birth_n'])
        n_item_birth_n = int((np.random.rand() * (float(self.param['Max_item_birthrate']) - float(self.param['Min_item_birthrate'])) + float(self.param['Min_item_birthrate'])) * len(self.item_list))
#        stage_rate = np.zeros(4)
#        for item in self.item_list:
#            stage_rate[int(item.x['stage']) + 1] += 1
#        for j in range(0, len(stage_rate)):
#            stage_rate[j] = stage_rate[j] / len(self.item_list)
#
#        n_item_birth_n = 1 if self.item_dead_count==0 else self.item_dead_count + int(100.0 * ( float(self.param['startup_item_rate']) - stage_rate[1]))
#        item_birth = np.random.rand(n_item_birth_n, 1)
        n_item_birth_n = 1 if self.item_dead_count == 0 else self.item_dead_count

        for j in range(0, n_item_birth_n):
            it = Item()
            self.id_count += 1
            it.init_random(self.param, self.im, t0=float(t), id=self.id_count)
            self.item_list.append(it)
            self.item_new_count += 1
            #rank
    def item_death_decline(self, t):
        'decline death items accoding to ipv and life stage'
        self.item_dead_count = 0
        for item in self.item_list:
            if item.x['stage'] == -1:
               self.item_list.remove(item)
               self.item_dead_count += 1

    def observe_item(self, item, t):
        """
        获取商品表示
        :param item: 
        :param t: 
        :return: 
        """
        return item.normalized_observe(t)

        t_dq = scale2limit(item.x['dq'], 1e3, 1e5, 0, 1)
        t_dt = scale2limit(item.x['t'] - item.x['t0'], 1e1, 1e2, 0, 1)
        t_q = scale2limit(item.x['q'], 1e4, 1e7, 0, 1)
        t_ctr = scale2limit(item.x['ctr'], 0.0, 0.2, 0, 1)
        t_tlc = scale2limit(item.x['theta_lower_ctr'], 0.0, 0.2, 0, 1)
        t_thc = scale2limit(item.x['theta_higher_ctr'], 0.0, 0.2, 0, 1)
        x = [t_dq, t_dt, t_q, t_ctr, t_tlc, t_thc]
#        x1 =  [ log(log(item.x['dq'] + 1.0) + 0.001), log(log(item.x['t'] - item.x['t0'] + 1.0) + 0.001), log(log(item.x['q'] + 1.0) + 0.001), item.x['ctr']]
#        x2 = [ item.x['theta_lower_ctr'], item.x['theta_higher_ctr']]
        #x = x1 + x2
#        self.param['dim1_action'] = 4
        return x

    def get_action_dim(self):
       return int(self.param['dim1_action'])

    def get_state_dim(self):
       return int(self.param['dim1_state'])

    def action(self, a, t):
        """
        1、商品的生成和消亡控制
        2、执行a动作各商品分配到的流量及商品自身状态的更新（生命周期、ctr）
        :param a: 
        :param t: 
        :return: 
        """
        # Model a ranker of the taobao platform
        self.item_birth_grow(t)
        self.item_death_decline(t)
        w = a
        sum_s = 0.0
        # 每轮的总流量
        Q_all = float(self.param['Q_all'])
        for item in self.item_list:
            # todo:目前的状态表示很弱
            x = self.observe_item(item, t)
            s = 1.0 / (1.0 + exp( - float(np.dot(x, a))))
            item.x['score'] = s
            sum_s += s
        for item in self.item_list:
            dq = Q_all * item.x['score'] / sum_s
            item.update_status(dq, t)
        self.record_origin_action(t, a)

    def step(self, a):
        """
        执行action a之后的状态更新和reward
        :param a: 
        :return: 
        """
        self.t += float(self.param['step_dT'])
        self.action(a, self.t)
        next_state, reward, info = self.observe(self.t)
        self.record_step(self.t, next_state[0], a, reward, info)
        return next_state, reward, info

    def observe_minmax(self, t):
        pass

    def observe_04(self, t):
        x, reward, info = self.observe_03(t)
        x = [item.reshape(int(self.param['dim1_item_state']), int(self.param['dim1_state'])//int(self.param['dim1_item_state'])).T \
             for item in x]
        return x, reward, info

    def pca_sorted(self, sublist, t):
        m = int(self.param['dim1_item_state'])
        n = int(len(sublist))
        X = np.ndarray(shape=(n,m))
        for j in range(0, n):
            item = sublist[j]
            xnow = item.normalized_observe(t)
            X[j,:] = xnow[:]
        X -= np.mean(X, axis=0)
        pca = PCA(n_components=1)
        pca.fit_transform(X)
        p1 = np.transpose(pca.components_)
        if t < int(self.param['initial_stage_step']):
            self.p0 = p1
            self.pca_tau=float(self.param['pca_tau'])
        else:
            err1 = np.linalg.norm(p1 - self.p0)
            err2 = np.linalg.norm(-p1 - self.p0)
            if err1 > err2:
                p1 = - p1
            err = np.linalg.norm(p1 - self.p0)
            tau = self.pca_tau  
            # if err > 0.3:
            if err > 1.0:
                print('norm: ', np.linalg.norm(p1), np.linalg.norm(self.p0))
                print('warning! pca changes too frequently! err=', err, ' p1= ', np.transpose(p1), ' p0= ', np.transpose(self.p0))
                tau = 0.01
                
            nr = np.linalg.norm((1.0 - tau) * self.p0 + tau * (p1 - self.p0))
            ptmp = ((1.0 - tau) * self.p0 + tau * (p1 - self.p0))/nr
            self.p0 = ptmp
        #print(np.transpose(p1))
        #print(np.transpose(self.p0))

        y = np.dot(X, self.p0)
        for j in range(0, n):
            sublist[j].x['pca_scalar'] = y[j]
        return sorted(sublist, key=lambda item: item.x['pca_scalar'])

    def observe_03(self, t):
        # concat the sampled data and makeup a constant length of state vector as state input
        x = np.zeros(int(self.param['dim1_state']))
        xl = []
        m = int(self.param['dim1_item_state'])
        n = int(int(self.param['dim1_state']) / m)
        if len(self.item_list) < n:
            print('error! number of items in item_list is small than sample! len(list)=%d, n=%d' % (
            len(self.item_list), n))
            return None
        for k in range(int(self.param['sampling_times'])):
            sublist = random.sample(self.item_list, n)
            # arrange order in of the list
            if int(self.param['valid_pca'])==1:
                sublist = self.pca_sorted(sublist, t)
            else:
                sublist = sorted(sublist, key=lambda item: item.x['t'] - item.x['t0'])

            for j in range(0, n):
                item = sublist[j]
                xnow = item.normalized_observe(t)
                #xnow = item.observe(t)
                x[j * m: (j + 1) * m] = xnow[:]
            xl.append(x)
        reward = self.get_reward(t)
        info = None
        self.record_all_item(t)
        self.record_metrics(t, xl[0], reward, info)
        self.record_state(t, x)

        return xl, reward, info

    def observe_02(self, t):
        # concat the sampled data and makeup a constant length of state vector as state input
        x = np.zeros(int(self.param['dim1_state']))
        m = int(self.param['dim1_item_state'])
        n = int(self.param['dim1_state']) // m
        if len(self.item_list) < n :
            print('error! number of items in item_list is small than sample! len(list)=%d, n=%d'%(len(self.item_list), n))
            return None
        sublist = random.sample(self.item_list, n)
        # arrange order in of the list 
        sorted(sublist, key = lambda item : item.x['t'] - item.x['t0'])
        for j in range(0, n):
            item = sublist[j]
            xnow = item.observe(t)
            x[ j * m : (j + 1) * m ] = xnow[:]
        reward = self.get_reward(t)
        info = None
#        self.record_all_item(t)
        self.record_metrics(t, x, reward, info)
        return x, reward, info

    def observe_01(self, t):
        x = np.zeros(int(self.param['dim1_state']))
        xl = []
        s_cnt = len(self.item_list)
        s_x = np.zeros(shape=(len(self.item_list), 4))

        for j in range(s_cnt):
            item = self.item_list[j]
            s_x[j, 0] = item.x['t'] - item.x['t0']
            s_x[j, 1] = item.x['ctr']
            s_x[j, 2] = item.x['q']
            s_x[j, 3] = 1 if item.x['stage'] == 0 else 0 

        x[0] = np.mean(s_x[:, 1]) # 平均ctr
        x[1] = np.mean(s_x[:, 2]) # 平均流量
        x[2] = np.std(s_x[:, 1])  # ctr的标准差
        x[3] = np.mean(s_x[:, 3]) # 新品占比
        x[4] = np.mean(s_x[:, 0]) # 平均商品启动时间

        xl.append(x)
        reward =  self.get_reward(t)
        info = None

        #self.record_all_item(t)
        self.record_metrics(t, xl[0], reward, info) 

        return xl, reward, info

    def observe(self, t):
        # observe only
        if int(self.param['valid_2d_obs'])==1:
            return self.observe_04(t)
        elif int(self.param['dim1_state']) > 100 :
            return self.observe_03(t)
        else:
            return self.observe_01(t)       

    def record_all_item(self, t):
        if int(self.param['record']) != 1:
            return
        line = '%f' % (t)
        n = 100
        sub_list = []
        zero_item = Item()
        zero_item.init_random(self.param, self.im, t0=float(t), id=0)
        for val in zero_item.x:
            zero_item.x[val] = 0
        for k in range(20):
            found = False
            found_item = zero_item
            for item in self.item_list:
                if (int(item.x['id']) // 1000) == k and int(item.x['id']) % 1000 == 0:
                    found_item= item
                    found = True
            sub_list.append(found_item)
        sub_list += random.sample(self.item_list, n)
        for item in sub_list:
            line += ' %d %f %f %f %f %f %f %f' % (
            item.x['id'], item.x['t'] - item.x['t0'], item.x['q'], item.x['dq'], item.x['ctr'],
            item.x['theta_lower_ctr'], item.x['theta_higher_ctr'], float(item.x['stage']))
            for k in range(int(self.param['im_feature_dim'])):
                line += ' %f' % (item.x['feature_' + str(k)])
        self.fp_record_item.write(line + '\n')

    def record_metrics(self, t, x, reward, info):
        if int(self.param['record']) != 1:
            return 
        s_Q = 0
        s_ctr = 0
        s_cnt = 0
        i_id = 120
        i_t = 0
        i_q = 0
        i_ctr = 0
        s_grow_up_cost_time_all = 0
        s_grow_up_cost_time_cnt = 0
        s_stage_cnt = np.zeros(5)
        s_stage_score = np.zeros(5)
        s_stage_dq = np.zeros(5)
        s_stage_q = np.zeros(5)
        s_stage_ctr = np.zeros(5)
        for item in self.item_list:
           s_Q += item.x['dq']
           s_ctr += item.x['ctr']
           s_cnt += 1
           s_stage_cnt[item.x['stage'] + 1] += 1  
           s_stage_score[item.x['stage'] + 1] += item.x['score']
           s_stage_dq[item.x['stage'] + 1] += item.x['dq']
           s_stage_q[item.x['stage'] + 1] += item.x['q']
           s_stage_ctr[item.x['stage'] + 1] +=item.x['ctr']
           if item.x['id'] == i_id:
                i_t = item.x['t'] - item.x['t0']
                i_q = item.x['q']
                i_ctr = item.x['ctr']
           if item.x['stage'] == 2 and item.x['grow_up_cost_time'] < 1e4:
                s_grow_up_cost_time_all += item.x['grow_up_cost_time']
                s_grow_up_cost_time_cnt += 1
        for j in range(0, len(s_stage_score)):
            s_stage_score[j] = s_stage_score[j] / s_stage_cnt[j] if s_stage_cnt[j] > 0 else 1.0
            s_stage_dq[j] = s_stage_dq[j] / s_stage_cnt[j] if s_stage_cnt[j] > 0 else 1.0
            s_stage_q[j] = s_stage_q[j] / s_stage_cnt[j] if s_stage_cnt[j] > 0 else 1.0
            s_stage_ctr[j] = s_stage_ctr[j] / s_stage_cnt[j] if s_stage_cnt[j] > 0 else 1.0
        s2 =  0.0
        for item in self.item_list:
            s2 += item.x['dq'] * item.x['ctr'] #todo:应该使用动态ctr来计算点击
        line = '%f' % (t)
        line += ' %d' % (len(self.item_list))
        line += ' %d' % ((self.item_new_count))
        line += ' %d' % ((self.item_dead_count))
        line += ' %d %f %f %f' % (i_id, i_t, i_q, i_ctr)
        line += ' %f' % (s_ctr/s_cnt)
        line += ' %f' % (s_Q)
        line += ' %f' % (reward)
        line += ' %s' % (0.0)
        line += ' %f %f %f %f %f' % (x[0], x[1], x[2], x[3], x[4])
        for j in range(0, len(s_stage_score)):
            line += ' %f %f %f %f %f' % (s_stage_cnt[j], s_stage_score[j], s_stage_dq[j], s_stage_q[j], s_stage_ctr[j])
        line += ' %f' % (s2 / s_cnt)
        line += ' %f' %(s_grow_up_cost_time_all / s_grow_up_cost_time_cnt if s_grow_up_cost_time_cnt > 0 else 1.0)
        #print(line)
        self.fp_record_metrics.write(line + '\n')

    def record_origin_action(self, t, action):
        if int(self.param['record']) != 1:
            return
        line = '%f'%(t)
        for k in range(action.shape[0]):
            line += ' %f'%(action[k])
        self.fp_record_action.write(line + '\n')

    def record_state(self, t, x):
        if int(self.param['record']) != 1:
            return
        line = '%f'%(t)
        for j in range(min(len(x), int(self.param['record_state_max_num']))):
            line += ' %f'%(x[j])
        self.fp_record_state.write(line + '\n')

    def record_step(self, t, state, action, reward, info):
        if int(self.param['record']) != 1:
            return
        line =  'eps:%f'%(t)
        line += ' action:%s'%(np.array_str(action, max_line_width=500))
        line += ' reward:%f'%(reward)
        line += ' info:%s'%(str(info))
        line += ' state:%s'%(np.array_str(np.array([state[0], state[1], state[2], state[3], state[4]]), max_line_width=500))
        self.fp_record_step.write(line + '\n')

    def save_env(self, filename):
        # save params
        fp = None
        try :
            fp = open(filename, 'w')
        except IOError:
            print('error! can not open file: %s for write!'%(fileanme))
            return -1
        for item in self.item_list:
            fp.write(item.state2str() + '\n')
        fp.close()

    def load_env(self, filename):
        # load item status
        fp = None
        try :
            fp = open(filename, 'r')
        except IOError:
            print('error! can not open file: %s for read!'%(filename))
            return -1
        self.t = 0.0
        self.id_count = 0
        self.item_list = []
        self.item_new_count = 0
        self.item_dead_count = 0
        for line in fp.readlines():
            item = Item()
            item.str2state(line.strip().replace('\n',''))
            self.item_list.append(item)
            self.id_count = max(self.id_count, int(item.x['id']))
         
    def init_random_env(self):
        """env初始化：构造商品集合，需要包含不同生命周期的商品，且商品的初始流量是不同的
        
        :return: 
        """
        n_item_init = int(self.param['n_item_init'])
        self.item_list = []
        self.item_new_count = 0
        self.item_dead_count = 0 
        self.id_count = 0
        self.t = 0.0
        for i in range(0, n_item_init):
            it = Item() 
            self.id_count += 1
            # 商品状态初始化
            it.init_random(self.param, self.im, t0=0.0, id=self.id_count)
            # 不同生命周期的商品比例控制，并更新ctr
            it.random_init_stages(t=0.0)
            self.item_new_count += 1 if it.x['stage'] == 0 else 0
            self.item_list.append(it)

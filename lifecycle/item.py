# coding=utf-8
from __future__ import print_function
import numpy as np
from math import *
from item_nn_model import ItemModel

#     def init_random(self, param, t0=0, id=0):
#     def softmax1d(self, x, x_min, x_max, y_min, y_max):
#     def f(self, u, t):
#     def update_status(self, dq, t):
#     def observe(self, t):
#     def random_init_stages(self, t=0):
#     def state2vec(self):
#     def state2str(self):
#     def str2state(self, line):
#     def manual_set_status(self, t0=0, t=0, stage=0, q=0, ctr=0, dq=0):
class Item(object):
    def __init__(self):
        self.x = {}

    def init_random(self, param, im, t0=0, id=0):
        """item初始化
        
        :param param: 
        :param t0: 
        :param id: 
        :return: 
        """
        # 启动完成需要的累计流量
        self.x['q_startup'] = np.random.random() * (
            float(param['Max_q_startup']) - float(param['Min_q_startup'])) + float(param['Min_q_startup'])
        # 启动完成的时间
        self.x['time_startup2decline'] = np.random.random() * (
            float(param['Max_time_startup2decline']) - float(param['Min_time_startup2decline'])) + float(
            param['Min_time_startup2decline'])
        # 启动期持续时间
        self.x['time_startup'] = np.random.random() * (
            float(param['Max_time_startup']) - float(param['Min_time_startup'])) + float(param['Min_time_startup'])
        # 成熟期持续时间
        self.x['time_maturity'] = np.random.random() * (
            float(param['Max_time_maturity']) - float(param['Min_time_maturity'])) + float(param['Min_time_maturity'])
        # 衰退期持续时间
        self.x['time_decline'] = np.random.random() * (
            float(param['Max_time_decline']) - float(param['Min_time_decline'])) + float(param['Min_time_decline'])
        # ctr
        self.x['theta_lower_ctr'] = np.random.random() * (
            float(param['Max_theta_lower_ctr']) - float(param['Min_theta_lower_ctr'])) + float(
            param['Min_theta_lower_ctr'])
        self.x['theta_higher_ctr'] = np.random.random() * (
            float(param['Max_theta_higher_ctr']) - float(param['Min_theta_higher_ctr'])) + float(
            param['Min_theta_higher_ctr'])
        xf = np.random.rand(int(param['im_feature_dim']))
        yf = im.normal_predict(xf.reshape(1,int(param['im_feature_dim'])))[0,:] 
        for k in range(int(param['im_feature_dim'])):
            self.x['feature_'+str(k)] = xf[k]
        self.x['theta_lower_ctr'] = im.scale2normal(yf[0], float(param['Min_theta_lower_ctr']), float(param['Max_theta_lower_ctr']))
        self.x['theta_higher_ctr'] = im.scale2normal(yf[1], float(param['Min_theta_higher_ctr']), float(param['Max_theta_higher_ctr']))
        self.x['growth_first_q'] = 0.0
        self.x['grow_up_cost_time']  = 1e9
        self.x['pca_scalar'] = 0.0
        self.x['click'] = 0.0
        # 商品上线的时间戳
        self.x['t0'] = t0
        # 当前ctr的默认值
        self.x['ctr'] = 0.0
        # 累计流量
        self.x['q'] = 0.0
        # 当前轮分配到的流量
        self.x['dq'] = 0.0
        # 当前时间戳
        self.x['t'] = self.x['t0']
        # t-t0
        self.x['dt'] = 0.0
        # 商品编号
        self.x['id'] = id
        # 生命周期
        self.x['stage'] = 0
        # 最新stage的开始时间
        self.x['lsct'] = t0
        #
        self.x['score'] = 0.0
        self.param = param
        # 根据商品所属的生命周期更新商品的ctr，初始化的时候会被random_init_stages中的调用覆盖，但在整个实验过程中会陆续有一些新品加入，所以此处的调用是必须的
        self.update_status(0.0, float(t0))

    def softmax1d(self, x, x_min, x_max, y_min, y_max):
        if abs(y_min - y_max) < 1e-6:
            return y_min
        if y_min < y_max:
            x = x_min if x < x_min else x
            x = x_max if x > x_max else x
            tx = (x - x_min) / (x_max - x_min) * 8.0 - 4.0
            ty = 1.0 / (1.0 + exp(-tx))
            y = (ty - 1.0 / (1 + exp(4.0))) / (1.0 / (1.0 + exp(-4.0)) - 1.0 / (1.0 + exp(4.0))) * (
                y_max - y_min) + y_min
        else:
            x = x_min if x < x_min else x
            x = x_max if x > x_max else x
            tx = (x - x_min) / (x_max - x_min) * 8.0 - 4.0
            ty = 1.0 - 1.0 / (1.0 + exp(-tx))
            y = (ty - 1.0 + 1.0 / (1.0 + exp(-4.0))) / (1.0 / (1.0 + exp(-4.0)) - 1.0 / (1.0 + exp(4.0))) * (
                y_min - y_max) + y_max
        return y

    def f2(self, u, t):
        """Generate ctr for item at different stage
        
        :param u: u[0] for dq
        :param t: iteration 
        :return: 
        """
        if self.x['stage'] == -1:
            return

        self.x['dq'] = u[0]
        self.x['t'] = t
        self.x['dt'] = self.x['t'] - self.x['t0']
        self.x['q'] = self.x['q'] + self.x['dq']
        t = self.x['t']
        dq = self.x['dq']
        q = self.x['q']
        z0 = self.x['stage']
        ctr = 0
        dt = t - self.x['lsct']
        # todo:added at 2018.1.15
        ## 新品的ctr与累计流量和启动所需的累计流量阈值有关
        ## 成熟期商品的ctr只与持续时间有关，感觉不太严谨，直觉上商品的效率是影响ctr的主要因素
        ## 衰退期商品的ctr也有类似的问题
        if z0 == 0:  # stage growth
            ctr = self.softmax1d(q, 0, self.x['q_startup'], self.x['theta_lower_ctr'], self.x['theta_lower_ctr'])
        elif z0 == 1:
            ctr = self.softmax1d(q, 0, self.x['q_startup'], self.x['theta_lower_ctr'], self.x['theta_higher_ctr']) 
        elif z0 == 2:  # stage maturity
            ctr = self.softmax1d(dt, 0.0, self.x['time_maturity'], self.x['theta_higher_ctr'], self.x['theta_higher_ctr'])
        elif z0 == 3:  # stage decline
            ctr = self.softmax1d(dt, 0.0, self.x['time_decline'], self.x['theta_higher_ctr'], self.x['theta_lower_ctr'])
        else:
            print('state error')
            return
        # todo:added at 2018.1.15
        ## ctr上的噪声影响有多大
        ctr = ctr + (np.random.rand() - 0.5) * ((self.x['theta_higher_ctr'] - self.x['theta_lower_ctr'])) * float(
            self.param['Max_ctr_noise_rate']) * 5.0
        # ctr = ctr + (0.723 - 0.5) * ((self.x['theta_higher_ctr'] - self.x['theta_lower_ctr']) * 0.5) * float(self.param['Max_ctr_noise_rate'])
        ctr = 0 if ctr < 0.0 else ctr
        ##ctr的范围应该更小一些
        ctr = 1.0 if ctr > 1.0 else ctr
        # lsct last state change time
        if z0 == 0 and dt > self.x['time_startup']:
            z0 = 1
            self.x['lsct'] = t
            self.x['growth_first_q'] = self.x['q']
        elif z0 == 1 and q - self.x['growth_first_q'] > self.x['q_startup']:
            z0 = 2
            self.x['grow_up_cost_time'] = t - self.x['lsct']
            self.x['lsct'] = t
        elif z0 == 2 and dt > self.x['time_maturity'] or (z0 == 1 and dt > self.x['time_startup2decline']):
            z0 = 3
            self.x['lsct'] = t
        elif z0 == 3 and dt > self.x['time_decline']:
            z0 = -1
            self.x['lsct'] = t
            ctr = 0
        else:
            pass

        self.x['stage'] = z0
        self.x['ctr'] = ctr
        self.x['click'] = self.x['click'] + self.x['dq'] * self.x['ctr']
    def f(self, u, t):
        """Generate ctr for item at different stage
        
        :param u: u[0] for dq
        :param t: iteration 
        :return: 
        """
        if self.x['stage'] == -1:
            return

        self.x['dq'] = u[0]
        self.x['t'] = t
        self.x['dt'] = self.x['t'] - self.x['t0']
        self.x['q'] = self.x['q'] + self.x['dq']
        t = self.x['t']
        dq = self.x['dq']
        q = self.x['q']
        z0 = self.x['stage']
        ctr = 0
        dt = t - self.x['lsct']
        # todo:added at 2018.1.15
        ## 新品的ctr与累计流量和启动所需的累计流量阈值有关
        ## 成熟期商品的ctr只与持续时间有关，感觉不太严谨，直觉上商品的效率是影响ctr的主要因素
        ## 衰退期商品的ctr也有类似的问题
        if z0 == 0:  # stage growth
            ctr = self.softmax1d(q, 0, self.x['q_startup'], self.x['theta_lower_ctr'], self.x['theta_higher_ctr'])
        elif z0 == 1:  # stage maturity
            ctr = self.softmax1d(dt, 0.0, self.x['time_maturity'], self.x['theta_higher_ctr'],
                                 self.x['theta_higher_ctr'])
        elif z0 == 2:  # stage decline
            ctr = self.softmax1d(dt, 0.0, self.x['time_decline'], self.x['theta_higher_ctr'], self.x['theta_lower_ctr'])
        else:
            print('state error')
            return
        # todo:added at 2018.1.15
        ## ctr上的噪声影响有多大
        ctr = ctr + (np.random.rand() - 0.5) * ((self.x['theta_higher_ctr'] - self.x['theta_lower_ctr'])) * float(
            self.param['Max_ctr_noise_rate'])
        # ctr = ctr + (0.723 - 0.5) * ((self.x['theta_higher_ctr'] - self.x['theta_lower_ctr']) * 0.5) * float(self.param['Max_ctr_noise_rate'])
        ctr = 0 if ctr < 0.0 else ctr
        ##ctr的范围应该更小一些
        ctr = 1.0 if ctr > 1.0 else ctr
        # lsct last state change time
        if z0 == 0 and q > self.x['q_startup']:
            z0 = 1
            self.x['lsct'] = t
        elif z0 == 1 and dt > self.x['time_maturity'] or (z0 == 0 and dt > self.x['time_startup2decline']):
            z0 = 2
            self.x['lsct'] = t
        elif z0 == 2 and dt > self.x['time_decline']:
            z0 = -1
            self.x['lsct'] = t
            ctr = 0
        else:
            pass

        self.x['stage'] = z0
        self.x['ctr'] = ctr

    def update_status(self, dq, t):
        """更新商品的ctr和stage
        
        :param dq: 
        :param t: 
        :return: 
        """
        u = np.zeros(2)
        u[0] = np.array(float(dq))
        self.f2(u, t)

    def observe(self, t):
        return self.state2vec()

    def normalized_observe(self, t):
        return self.normalized_state2vec()

    def random_init_stages(self, t=0):
        """
        不同生命周期的商品比例的控制，并且设置不同新品的流量初值不同
        :param t: 
        :return: 
        """
        if np.random.rand() < float(self.param['startup_item_rate']):
            q = np.random.rand() * (self.x['q_startup'])
            self.manual_set_status(t0=self.x['t0'], t=t, stage=0, q=q, ctr=0, dq=0)
        else:
            t0 = t - np.random.rand() * self.x['time_maturity']
            # todo:added at 2018.15
            ##非新品就都是成熟期了么？
            self.manual_set_status(t0=t0, t=t, stage=1, q=self.x['q_startup'], ctr=0, dq=0)

    def scale2limit(self, x, xmin, xmax, ymin, ymax):
        ans = (x - xmin)/ (xmax - xmin) * (ymax - ymin) + ymin
        ans = ans if ans > ymin else ymin
        ans = ans if ans < ymax else ymax
        return ans

    def normalized_state2vec(self):

        x1 = np.zeros(8 + int(self.param['im_feature_dim']))
        x1[0] = 1.0 if self.x['stage']==0 else -1.0
        x1[1] = 1.0 if self.x['stage']==1 else -1.0
        x1[2] = 1.0 if self.x['stage']==2 else -1.0
        x1[3] = 1.0 if self.x['stage']==3 else -1.0
        x1[4] = self.scale2limit(self.x['ctr'], 0.0, 0.2, -1.0, 1.0)
        x1[5] = self.scale2limit(self.x['q'], 1e4, 1e7, -1.0, 1.0)
        x1[6] = self.scale2limit(self.x['dq'], 5e3, 2e4, -1.0, 1.0)
        x1[7] = self.scale2limit(self.x['dt'], 1e1, 3e2, -1.0, 1.0)

        for k in range(int(self.param['im_feature_dim'])):
            x1[8 + k] = self.scale2limit(self.x['feature_'+str(k)], 0.0, 1.0, -1.0, 1.0)

        y = x1
        return y

    def state2vec(self):
        x1 = [self.x['id'], self.x['stage'], self.x['ctr'], self.x['q'], self.x['dq'], self.x['dt'], self.x['t']]
        x2 = []
        for k in range(int(self.param['im_feature_dim'])):
            x2 += [self.x['feature_'+str(k)]]
        x = x1 + x2 
        y = np.array(x)
        return y

    def state2str(self):
        line = ''
        for k, v in sorted(self.x.items(), key=lambda d: d[0]):
            if line == '':
                line = '%s:%s' % (k, str(v))
            else:
                line += ',%s:%s' % (k, str(v))
        return line

    def str2state(self, line):
        for item in line.split(','):
            k, v = item.split(':')
            self.x[k] = float(v)

    def manual_set_status(self, t0=0, t=0, stage=0, q=0, ctr=0, dq=0):
        self.x['t0'] = t0
        self.x['stage'] = stage
        self.x['q'] = q
        self.x['dq'] = dq
        self.x['ctr'] = ctr
        self.x['lsct'] = t0
        self.update_status(dq, t)


if __name__ == "__main__":
    # test item state
    # test ctr model: plot 2 different items, and feed PV to the item, observe the ctr changes
    # test : state2str, str2state 
    filepath_param = 'param'
    param = {}
    with open(filepath_param, 'r') as f:
        for line in f:
            k, v = line.strip().replace('\n', '').split(':')
            param[k] = v
    print('param: ', param)
    #    np.random.seed(int(param['random_seed']))
    im = ItemModel(param)
    np.random.seed(29)
    it = Item()
    it.init_random(param, im, 0, 0)
    print('id = ', it.x['id'], ' x = ', it.state2str())
    ans = []
    q = 0
    for t in range(0, 180, 1):
        dq = 5e4
        #        dq = 5e4 + np.random.rand() * 5e4
        q += dq
        it.update_status(dq, t)
        ans.append([q, t, it.x['ctr'], it.x['q'], it.x['stage']])
        if t == 90:
            state_str = it.state2str()
    with open(param['filepath_tmp_result'], 'w') as f:
        print('ans length:', len(ans))
        n = len(ans)
        for i in range(0, n):
            f.write('%d %d %lf %d %d' % (ans[i][0], ans[i][1], ans[i][2], ans[i][3], ans[i][4]) + '\n')
    print('str state: ', it.state2str())
    line = 'lsct:98,theta_lower_ctr:0.0507447151839,theta_higher_ctr:0.104229687401,ctr:0,t0:0,time_maturity:34.395383285,t:98,q:7464942.46576,time_startup2decline:47.0943579137,score:0.0,q_startup:865122.385714,dt:0.0,stage:-1,time_decline:52.8971161299,id:0,dq:79628.8973321'
    it.str2state(line)
    print('str state: ', it.state2str())
    it.str2state(state_str)
    print([0.0, 90.0, it.x['ctr'], it.x['q'], it.x['stage']])
    for t in range(91, 93, 1):
        dq = 5e4
        #        dq = 5e4 + np.random.rand() * 5e4
        q += dq
        it.update_status(dq, t)
        print([q, t, it.x['ctr'], it.x['q'], it.x['stage']])

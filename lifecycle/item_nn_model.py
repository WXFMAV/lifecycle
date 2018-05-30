import tensorflow as tf 
import numpy as np
import math

def create_network(feature_dim, output_dim, layer_size = (48, 24)):
    #create network
    layer1_size = layer_size[0]
    layer2_size = layer_size[1]
    lbd = -1.0/math.sqrt(layer1_size)
    ubd = 1.0/math.sqrt(layer1_size)
    lbd3 = -3.0e-3
    ubd3 = 3.0e-3
    W1 = tf.Variable(tf.random_uniform([feature_dim, layer1_size], lbd, ubd), name='W1')
    b1 = tf.Variable(tf.random_uniform([layer1_size], lbd, ubd), name='b1')
    W2 = tf.Variable(tf.random_uniform([layer1_size, layer2_size], lbd, ubd), name='W2')
    b2 = tf.Variable(tf.random_uniform([layer2_size], lbd, ubd), name='b2')
    W3 = tf.Variable(tf.random_uniform([layer2_size, output_dim], lbd3, ubd3), name='W3')
    b3 = tf.Variable(tf.random_uniform([output_dim], lbd3, ubd3), name='b3')
        
    feature_input = tf.placeholder("float", [None, feature_dim])
    layer1 = tf.nn.relu(tf.matmul(feature_input, W1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    layer3 = tf.nn.tanh(tf.matmul(layer2, W3) + b3)
    network_output = layer3 
    net = [W1, b1, W2, b2, W3, b3]
    return feature_input, network_output, net

def create_multitask_network(input_dim, output_dim, layer_size = (48, 48, 24, 12)):
    #create network
    layer1_size = layer_size[0]
    layer2_size = layer_size[1]
#    lbd = -1.0/math.sqrt(layer1_size)
#    ubd = 1.0/math.sqrt(layer1_size)
    lbd = -0.5
    ubd = 0.5
    W1 = tf.Variable(tf.random_uniform([input_dim, layer1_size], lbd, ubd), name='W1')
    b1 = tf.Variable(tf.random_uniform([layer1_size], lbd, ubd), name='b1')
    W2 = tf.Variable(tf.random_uniform([layer1_size, layer2_size], lbd, ubd), name='W2')
    b2 = tf.Variable(tf.random_uniform([layer2_size], lbd, ubd), name='b2')
        
    x = tf.placeholder("float", [None, input_dim])
    layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    layer2 = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
    yshared = layer2
    net_shared = [W1, b1, W2, b2]
    y = []
    yk = []
    loss = []
    opt = []
    net = net_shared
    for k in range(output_dim):
        layer3_size = layer_size[2]
        layer4_size = layer_size[3]
        layer5_size = 1
        lbd3 = -1.0/math.sqrt(layer3_size) if 0 else -0.1
        ubd3 = 1.0/math.sqrt(layer3_size) if 0 else 0.1
        lbd4 = -1.0/math.sqrt(layer4_size) if 0 else -0.1
        ubd4 = 1.0/math.sqrt(layer4_size) if 0 else 0.1
        lbd5 = -1.0/math.sqrt(layer5_size) if 0 else -1 
        ubd5 = 1.0/math.sqrt(layer5_size) if 0 else 1
        W3 = tf.Variable(tf.random_uniform([input_dim, layer3_size], lbd3, ubd3), name='W3' + str(k))
        b3 = tf.Variable(tf.random_uniform([layer3_size], lbd3, ubd3), name='b3'+str(k))
        W4 = tf.Variable(tf.random_uniform([layer3_size, layer4_size], lbd4, ubd4), name='W4' + str(k))
        b4 = tf.Variable(tf.random_uniform([layer4_size], lbd4, ubd4), name='b4' + str(k))
        W5 = tf.Variable(tf.random_uniform([input_dim, layer5_size], lbd4, ubd4), name='W5' + str(k))
        b5 = tf.Variable(tf.random_uniform([layer5_size], lbd5, ubd5), name='b5' + str(k))
 
        W5c = tf.constant((np.random.rand(input_dim, 1) * 1.0/math.sqrt(input_dim)).astype(np.float32))
        b5c = tf.constant((np.random.rand(1) * 1e-2).astype(np.float32))
        W5.assign(W5c)
        b5.assign(b5c)

        layer3 = tf.nn.relu(tf.matmul(x, W3) + b3)
        layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
#        layer5 = tf.nn.tanh(tf.matmul(layer4, W5) + b5)
 #       layer4 = tf.matmul(x, W4) + b4
#        layer5 = tf.nn.tanh(tf.matmul(x, W5) + b5)
        layer5 = tf.matmul(x, W5) + b5
        y.append(tf.placeholder("float", [None, 1]))
        yk.append(layer5)
        loss.append(tf.reduce_mean(tf.reduce_sum(tf.square(y[k] - yk[k]), reduction_indices=[1])))
        #loss.append(tf.nn.l2_loss(y[k] - yk[k]))
        #opt.append(tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss[k]))
        #opt.append(tf.train.AdagradOptimizer(learning_rate = 0.1).minimize(loss[k]))
        #opt.append(tf.train.MomentumOptimizer(learning_rate = 0.1).minimize(loss[k]))
        opt.append(tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss[k]))

        net = net + [W3, b3, W4, b4, W5, b5]
    yall = tf.concat(yk, 1, 'concat')
    
    return x, y, yall, loss, opt, yshared, net

class ItemModel(object):
    def __init__(self, param):
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth=True  
        self.sess = tf.Session(config=config)
        self.param = param
        x, y, yall, loss, opt, yshared, net = create_multitask_network(int(param['im_feature_dim']), int(param['im_output_dim']))
        self.feature_input = x
        self.network_output = yall
        self.net = net
        for v in tf.trainable_variables():
            print(v)
        self.sess.run(tf.variables_initializer(self.net))
        self.set_min_max(1)
        self.restore_para()

    def normal_predict(self, feature):
        res = self.sess.run(self.network_output, feed_dict={self.feature_input:feature})
        res = (res - self.output_min) / (self.output_max - self.output_min) * 2.0 - 1.0
        return res
    def scale2normal(self, u, xmin, xmax):
        ans = (u + 1.0)/2.0 * (xmax - xmin) + xmin
        ans = ans if ans > xmin else xmin
        ans = ans if ans < xmax else xmax
        return ans
    def set_min_max(self, restore = 0):
        if restore == 1:
            f = open(self.param['filepath_im_param'], 'r')
            lines = f.readlines()
            self.output_min = np.array(lines[0].strip('\n').split(',')[0:int(self.param['im_output_dim'])]).astype(np.float32)
            self.output_max = np.array(lines[1].strip('\n').split(',')[0:int(self.param['im_output_dim'])]).astype(np.float32)
            f.close()
            return 
        n = 1000000
        sample = np.random.rand(n, int(self.param['im_feature_dim']))
        y = im.predict(sample.reshape(n, int(self.param['im_feature_dim'])))
        self.output_min = y.min(0)  
        self.output_max = y.max(0)
        f = open(self.param['filepath_im_param'], 'w') 
        line = ''
        for j in range(0, len(self.output_min)):
            line += str(self.output_min[j]) + ','
        f.write(line + '\n')    
        line = ''
        for j in range(0, len(self.output_max)):
            line += str(self.output_max[j]) + ','
        f.write(line + '\n')    
        f.close() 

    def predict(self, feature):
        
        return self.sess.run(self.network_output, feed_dict={self.feature_input:feature})
	
    def save_para(self):
        tf.train.Saver(self.net).save(self.sess, self.param['filepath_im_model'])
        

    def restore_para(self):
        tf.train.Saver(self.net).restore(self.sess, self.param['filepath_im_model'])

def loadparams():
    param = {}
    filepath_param = 'param'
    with open(filepath_param, 'r') as f:
        for line in f:
            k, v = line.strip().replace('\n','').split(':')
            param[k] = v
    print(param)
    return param

def generate_samples(param):
    n = 10000
    x = np.random.rand(n, int(param['im_feature_dim']))
    f_b = np.random.rand(1, int(param['im_feature_dim']))
    f_w = np.random.rand(int(param['im_output_dim']), int(param['im_feature_dim'])) * 100.0
    f_u = np.random.rand(1, int(param['im_output_dim']))
    y = np.zeros(shape=(n, int(param['im_output_dim'])))
    for j in range(n):
        y[j, :] = np.dot(f_w, ((x[j, :] - f_b) ** 2).reshape(int(param['im_feature_dim']),1)).reshape(int(param['im_output_dim']))
    
    return x, y

def train_model(s, param):
    m = int(param['im_output_dim'])
    n = int(param['im_feature_dim'])
    N = np.shape(s)[0]
    print(m, n, N)
    
    it = ItemModel(param)
 
#    x, yall, net = create_network(int(param['im_feature_dim']), int(param['im_output_dim']))
#    y = tf.placeholder('float', [None, 1])
#    loss = tf.nn.l2_loss(y - yall) 
#    opt = tf.train.AdamOptimizer().minimize(loss)
    #x, yall, net = create_network(int(param['im_feature_dim']), int(param['im_feature_dim']))
    x, y, yall, loss, opt, yshared, net = create_multitask_network(int(param['im_feature_dim']), int(param['im_output_dim']))
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    y_batch = s[:, 0:int(param['im_output_dim'])]
    x_batch = s[:, int(param['im_output_dim']):int(param['im_output_dim'])+int(param['im_feature_dim'])+1]
#    for iters in range(int(param['im_output_dim']) * 5):
#        k = int(np.random.rand() * int(param['im_output_dim']))
    for k in range(int(param['im_output_dim'])):
        _, yloss = sess.run([opt[k], loss[k]], feed_dict = {
                    x:x_batch,
                    y[k]:y_batch[:, k].reshape(N, 1)
                    })
        print(k, yloss)
    yans = sess.run(yall, feed_dict={x:x_batch})
    
    print('yans:', yans)
    res = np.concatenate((yans, y_batch, x_batch), axis = 1) 
    f = open(param['ckp_dir'] + '/model_test.txt', 'w')
    for k in range(np.shape(res)[0]):
        line = ''
        for j in range(np.shape(res)[1]):
            line += ' %f'%(res[k][j])
        f.write(line + '\n')
    f.close()

if __name__ == '__main__':
    param = loadparams() 
    im = ItemModel(param)
    n = 10000
    np.random.seed(4)
#    im.sess.run(tf.initialize_variables(im.net))
    for p in im.net:
        print(p.name)
#    im.set_min_max(1)
#    im.save_para()
#    im.restore_para()
    sample = np.random.rand(n, int(param['im_feature_dim']))
    y = im.predict(sample.reshape(n, int(param['im_feature_dim'])))
    y = im.normal_predict(sample.reshape(n, int(param['im_feature_dim'])))
    print(y.min(0))
    print(y.max(0))
    res = np.concatenate((y, sample), axis = 1)
    f = open(param['ckp_dir']+'/model_test.txt', 'w')
    for k in range(np.shape(res)[0]):
        line = ''
        for j in range(np.shape(res)[1]):
            line += ' %f'%(res[k][j])
        f.write(line + '\n')
    f.close()

    im.restore_para()
    y = im.predict(sample.reshape(n, int(param['im_feature_dim'])))
    #print sample
    #print y

#    x, y = generate_samples(param)
#    s = np.concatenate((y, x), axis = 1)
#    train_model(s, param)
#    print np.concatenate((y,x), axis=1)
       

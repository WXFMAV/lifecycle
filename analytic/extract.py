import tensorflow as tf
import numpy as np
import sys 
datpath = './dat/'
figpath = './fig/'
#fileevent = datpath + 'tf_event/events.out.tfevents.1516327737.tb2a05506.sqa.tbc'
fileevent = datpath + 'tf_event/dat_low_dimensions'

if len(sys.argv) == 4:
     datpath = sys.argv[1]
     figpath = sys.argv[2]
     fileevent = datpath + sys.argv[3]

fp = {}
fp['test_input/reward_avg'] = open(datpath + 'reward_avg.txt', 'w')
fp['test_training/critic_loss'] = open(datpath + 'critic_loss.txt', 'w')
fp['test_training/actor_loss'] = open(datpath + 'actor_loss.txt', 'w')

if __name__ == '__main__':
    step = 0
    for e in tf.train.summary_iterator(fileevent):
        step += 1
        for v in e.summary.value:
            if v.tag in fp:
                line = '%d %f'%(step, v.simple_value)
                fp[v.tag].write(line+'\n')
    if step % 1000 == 0:
        print('step =:', step)
for p in fp.values():
    p.close()

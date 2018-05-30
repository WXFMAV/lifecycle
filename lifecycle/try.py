import numpy as np
from math import * 
class Try01(object):
    def __init__(self):
        print('init')
    def softmax1d(self, x, x_min, x_max, y_min, y_max):
        if abs(y_min - y_max) < 1e-6:
            return y_min
        if y_min < y_max:
            x = x_min if x < x_min else x
            x = x_max if x > x_max else x
            tx = (x - x_min) / (x_max - x_min) * 8.0 - 4.0
            ty = 1.0 / (1.0 + exp(-tx))
            y = (ty - 1.0 / (1 + exp(4.0)))/(1.0 / (1.0 + exp(-4.0)) - 1.0 / (1.0 + exp(4.0))) * (y_max - y_min) + y_min
#            print('tx, ty, x, y: ', tx, ty, x, y)
        else:
            x = x_min if x < x_min else x
            x = x_max if x > x_max else x
            tx = (x - x_min) / (x_max - x_min) * 8.0 - 4.0
            ty = 1.0 - 1.0 / ( 1.0 + exp(-tx))
            y = (ty - 1.0 + 1.0 / ( 1.0 + exp(-4.0))) / (1.0 / (1.0 + exp(-4.0)) - 1.0 / (1.0 + exp(4.0))) * (y_min - y_max) + y_max
        return y
if __name__ == "__main__":
   try01 = Try01()  
   for i in range(-16, 17, 1):
        print i, try01.softmax1d(float(i), -16.0, 16.0, 0.0, 1.0)
 

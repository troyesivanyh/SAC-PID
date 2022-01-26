import time
import matplotlib
matplotlib.use("TkAgg")
import math


# no overshoot ,but adjustment time is long
class PID(object):
    def __init__(self,p=1, i=0, d=0):
        self.kp = p
        self.ki = i
        self.kd = d
        self.reset()

    def set(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd


    def reset(self):
        self.last_err=0
        self.last_last_err=0
        self.now_err=0
        self.output=0.0

    def update(self,error):
        self.now_err = error
        self.change_val = self.kp * (self.now_err - self.last_err) + self.ki *self.now_err + self.kd * (
                    self.now_err - 2 * self.last_err + self.last_last_err)
        self.last_last_err = self.last_err
        self.last_err = self.now_err
        self.output += self.change_val
        return self.output






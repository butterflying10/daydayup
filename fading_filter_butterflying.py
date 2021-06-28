# -*- coding: utf-8 -*-
# @prohject : 'filterpy-master'
# @Time    : 2021/6/24 10:28
# @Author  : Butterflying
# @Email   : 2668609932@qq.com
# @File    : fading_filter_butterflying.py
# code is far away from bugs with the god animal protecting
'''
渐消滤波
'''
from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape, array
import numpy.linalg as linalg
import warnings

import numpy.random as random
import matplotlib.pyplot as plt

from filterpy.common.discretization import Q_discrete_white_noise, Q_continuous_white_noise, block_diag


class FadingKalmanFilter(object):
    def __init__(self, alpha, dim_x, dim_z, dim_u=0):

        assert alpha >= 1
        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.alpha_ = alpha  # 渐消因子
        self.dim_x = dim_x  # 状态值个数
        self.dim_z = dim_z  # 观测值个数
        self.dim_u = dim_u  # 控制值个数

        self.x = zeros((dim_x, 1))  # Current state estimate 当前历元状态参数向量（dim_x*1）
        self.P = eye(dim_x)  # Current state covariance matrix 当前状态参数向量协方差阵（dim_x*dim_x）
        self.Q = eye(dim_x)  # Process noise covariance matrix 过程噪声协方差阵（dim_x*dim_x）
        self.B = 0.  # control transition matrix 控制矩阵（一般不考虑）
        self.F = np.eye(dim_x)  # state transition matrix  状态转移矩阵（dim_x*dim_x）
        self.H = zeros((dim_z, dim_x))  # Measurement function   观测设计矩阵（dim_z*dim_x）
        self.R = eye(dim_z)  # Measurement noise covariance matrix 观测噪声协方差矩阵（dim_z*dim_z）
        self.z = np.array([[None] * dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0  # kalman gain 增益矩阵
        self.y = zeros((dim_z, 1))  # residual are computed during the innovation step 新息向量
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty (measurement space)
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        #self._alpha_sq = 1.  # fading memory control
        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # Only computed only if requested via property 似然函数
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()  # 当前状态向量预测值
        self.P_prior = self.P.copy()  # 当前状态向量预测协方差阵

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()  # 当前状态向量估值
        self.P_post = self.P.copy()  # 当前状态向量估值的协方差阵

        self.inv = np.linalg.inv

    def settingalpha_(self,z, u=None, B=None, F=None, Q=None,H=None,P=None,R=None):
        #计算预测状态向量
        if B is None:
            B = self.B
        if P is None:
            P = self.P
        if R is None:
            R = self.R
        if H is None:
            H = self.H
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):  # 如果Q是标量
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            x = dot(F, self.x) + dot(B, u)
        else:
            x = dot(F, self.x)
        #计算预测残差向量
        y= z - dot(H, self.x)
        #计算预测残差向量的协方差阵
        S=self.alpha_*dot(y,y.T)/(1+self.alpha_)

        HF=dot(H,F)
        HFP=dot(HF,P)
        M=dot(HFP,HF.T)
        HQ=dot(H,Q)
        N=S-dot(HQ,H.T)-R


        alpha=np.trace(N)/np.trace(M)

        if alpha>=1:
            self.alpha_=alpha
        else:
            self.alpha_=1.




        pass
    # 预测
    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.控制向量（一般不用）

        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):  # 如果Q是标量
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q P变为状态预测向量的协方差矩阵
        self.P = self.alpha_ * dot(dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like 当前历元观测向量
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None 测量噪声矩阵
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None  设计矩阵
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            # z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # y = z - Hx 新息向量y
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        # S 新息向量y的协方差矩阵 R观测向量噪声的协方差矩阵
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        # K 增益矩阵
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # 当前历元的状态向量估值
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        # 状态向量估值的协方差矩阵
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

DO_PLOT = True
# 不动模型(一维)
def const__1d(alpha):

    dt = 1
    f = FadingKalmanFilter(alpha=alpha, dim_x=1, dim_z=1)
    #初始位置 0
    f.x = array([[0.]])
    f.F = array([[1.]
                 ])
    f.H = np.array([[1.]])
    f.P = np.eye(1)
    f.R = np.array([[100]])
    #f.Q=Q_discrete_white_noise(dim=2, dt=dt, var=1.)
    f.Q = np.array([[0.01]])  # process uncertainty
    return f

def test_const_vel():
    # 观测值
    m_p = []
    ideal_p = []
    with open("test_error2.txt", "r") as file:
        lines = file.readlines()
    for line in lines:
        line = line.split("  ")
        m_p.append(float(line[0]))
        ideal_p.append(float(line[2]))


    f= const__1d(1.)
    results_p = []
    error_p = []
    for i in range(0, 400):
        f.settingalpha_(array([m_p[i]]))
        print(f.alpha_)
        f.predict()
        f.update(array([m_p[i]]))

        # save data
        results_p.append(f.x[0, 0])
        error_p.append(f.x[0, 0] - ideal_p[i])
    if DO_PLOT:

        # p2, = plt.plot(m_p, 'r')
        p1, = plt.plot(error_p, 'b', alpha=0.5)
        # p5, = plt.plot(ideal_p, 'g')
        # plt.legend([p1, p2, p5],
        #            ["KF output", "measurement", "ideal"], loc=4)
        plt.grid(ls='--')
        plt.xlim((0,400))
        plt.ylabel('position error(m)')
        plt.xlabel('Time(s)')
        plt.show()

# 常加速度模型（一维）
def const_acc_1d(alpha):
    dt = 1.
    f = FadingKalmanFilter(alpha=alpha, dim_x=3, dim_z=2)
    # 初始位置为0，初始速度为5，初始加速度为0
    f.x = array([[0], [5.], [0.]])
    # 状态转移矩阵
    f.F = array([[1., dt, 0.5 * dt ** 2],
                 [0., 3., dt],
                 [0, 0., 1.]])
    # 设计矩阵
    f.H = np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    ])
    # 状态参数向量协方差阵 初始值设为单位阵
    f.P *= np.eye(3)
    # f.P=np.array([[]
    # ])
    # 观测噪声协方差矩阵 位置观测噪声方差是100，速度观测噪声方差是0.01
    f.R = np.array([[100., 0.],
                    [0., 0.01]])
    # 过程噪声矩阵
    f.Q = Q_continuous_white_noise(dim=3, dt=dt, spectral_density=1.)

    return f
#动力学模型是静止模型，但观测到的是常加速度模型
def generating_error_data(vel, pos, dt):
    m_p = []
    m_v = []
    ideal_p = []
    ideal_v = []
    ip = pos
    iv = vel
    # 第一阶段  匀加速运动0.05m/s2  200-400s
    for i in range(0, 400):

        # 理想值
        iv+=0.05*dt
        ip += iv * dt + 0.5 * 0.05 * dt ** 2
        ideal_p.append(ip)
        ideal_v.append(iv)

        pos = ip + random.randn() * 10.
        vel = iv + random.randn() * 0.1
        m_v.append(vel)
        m_p.append(pos)

    return m_p, m_v, ideal_p, ideal_v
    pass
def generating_data(vel, pos, dt):
    '''
    生成实验数据 （用于常加速度模型）
    Parameters
    ----------
    sqrt_R   该实验数据方差的根号值
    vel  速度
    acc 加速度
    pos 位置
    dt 时间间隔
    Returns
    -------
    '''
    m_p = []
    m_v = []
    ideal_p = []
    ideal_v = []
    ip = pos
    iv = vel
    # 第一阶段  匀速5m/s  1-200s
    for i in range(0, 200):
        # random.randn() 生成标准正态分布

        # vel += random.randn() * 0.01
        # m_v.append(vel)
        # pos = vel * dt*(i+1) + random.randn() * 5.
        # m_p.append(pos)
        # 理想值
        ip += iv * dt
        ideal_p.append(ip)
        ideal_v.append(iv)
        pos = ip + random.randn() * 10.
        vel = iv + random.randn() * 0.1
        m_v.append(vel)
        m_p.append(pos)

    # 第二阶段  匀加速运动0.05m/s2  200-400s
    for i in range(200, 400):
        # vel += 0.05 * dt + random.randn() * 0.01
        # pos += vel * dt + 0.5 * 0.05 * dt ** 2 + random.randn() * 5.
        # m_p.append(pos)
        # m_v.append(vel)

        # 理想值
        iv += 0.05 * dt
        ip += iv * dt + 0.5 * 0.05 * dt ** 2
        ideal_p.append(ip)
        ideal_v.append(iv)

        pos = ip + random.randn() * 10.
        vel = iv + random.randn() * 0.1
        m_v.append(vel)
        m_p.append(pos)

    # 第三阶段 匀减速运动 -0.02m/s2 400-600s
    for i in range(400, 600):
        # vel += -0.02 * dt + random.randn() * 0.01
        # pos += vel * dt - 0.5 * 0.02 * dt ** 2 + random.randn() * 5.
        #
        # m_p.append(pos)
        # m_v.append(vel)

        # 理想值
        iv += -0.02 * dt
        ip += iv * dt - 0.5 * 0.02 * dt ** 2
        ideal_p.append(ip)
        ideal_v.append(iv)

        pos = ip + random.randn() * 10.
        vel = iv + random.randn() * 0.1
        m_v.append(vel)
        m_p.append(pos)
    # 第四阶段 匀速运动  600-800s(真实)
    ep=ip
    ev=iv
    for i in range(600, 800):
        # random.randn() 生成标准正态分布

        # 理想值
        ip += iv * dt
        ideal_p.append(ip)
        ideal_v.append(iv)


        pos = ip + random.randn() * 10.
        vel = iv + random.randn() * 0.1
        m_v.append(vel)
        m_p.append(pos)
    #第五阶段  停止状态，但模型还是采用的是常加速度模型 （800-1000）
    for i in range(800, 1000):
        iv=0
        ideal_p.append(ip)
        ideal_v.append(iv)

        pos = ip + random.randn() * 10.
        vel = iv + random.randn() * 0.1
        m_v.append(vel)
        m_p.append(pos)

    return m_p, m_v, ideal_p, ideal_v
def test_const_acc():
    # 观测值
    m_p = []
    m_v = []
    ideal_p = []
    ideal_v = []
    with open("test_error.txt", "r") as file:
        lines = file.readlines()
    for line in lines:
        line = line.split("  ")
        m_p.append(float(line[0]))
        m_v.append(float(line[1]))
        ideal_p.append(float(line[2]))
        ideal_v.append(float(line[3]))


    f = const_acc_1d(1.)

    results_p = []
    results_v = []
    results_a = []
    error_p, error_v = [], []
    alpha_list=[]
    for i in range(0, 1000):
        f.settingalpha_(np.array([[m_p[i]], [m_v[i]]]))
        print(f.alpha_)
        alpha_list.append(f.alpha_)
        f.predict()
        f.update(np.array([[m_p[i]], [m_v[i]]]))

        # save data
        results_p.append(f.x[0, 0])
        error_p.append(f.x[0, 0] - ideal_p[i])
        results_v.append(f.x[1, 0])
        error_v.append(f.x[1, 0] - ideal_v[i])
        results_a.append(f.x[2, 0])

    if DO_PLOT:
        plt.subplot(411)
        # p2, = plt.plot(m_p, 'r')
        p1, = plt.plot(error_p, 'b', alpha=0.5)
        # p5, = plt.plot(ideal_p, 'g')
        # plt.legend([p1, p2, p5],
        #            ["KF output", "measurement", "ideal"], loc=4)
        plt.grid(ls='--')
        plt.ylabel('position error(m)')
        plt.xlim((0,1000))

        plt.subplot(412)
        # p4, = plt.plot(m_v, 'r')
        p3, = plt.plot(error_v, 'b', alpha=0.5)
        # p6, = plt.plot(ideal_v, 'g')
        plt.ylabel('velocity error(m/s)')
        # plt.legend([p3, p4, p6],
        #            ["KF output", "measurement", "ideal"], loc=4)
        plt.grid(ls='--')
        plt.xlim((0, 1000))
        plt.subplot(413)
        p7, = plt.plot(results_a, 'b', alpha=0.5)
        # p4, = plt.axhline(m_v, 'r')
        # p6, = plt.plot(ideal_v, 'g')
        plt.ylabel('acceleration(m/s^2)')
        # plt.legend([p7],
        #            ["KF output"], loc=4)
        plt.grid(ls='--')
        plt.xlim((0, 1000))
        plt.xlabel('Time(s)')
        plt.subplot(414)
        p8, = plt.plot(alpha_list, 'r', alpha=0.5)
        # p4, = plt.axhline(m_v, 'r')
        # p6, = plt.plot(ideal_v, 'g')
        plt.ylabel('alpha')
        # plt.legend([p7],
        #            ["KF output"], loc=4)
        plt.grid(ls='--')
        plt.xlim((0, 1000))
        plt.xlabel('Time(s)')
        plt.show()
if __name__ == "__main__":
    DO_PLOT = True
    # test_noisy_1d()
    # test_const_vel()
    #test_const_acc()
    # 生成数据
    # m_p, m_v, ideal_p, ideal_v = generating_data(5., 0., 1.)
    # with open("test_error.txt", "w") as file:
    #     for i in range(len(m_p)):
    #         file.write(str(m_p[i])+"  "+str(m_v[i])+"  "+str(ideal_p[i])+"  "+str(ideal_v[i])+'\n')
    # 生成错误——数据
    # m_p, m_v, ideal_p, ideal_v = generating_error_data(0., 0., 1.)
    # with open("test_error2.txt", "w") as file:
    #     for i in range(len(m_p)):
    #         file.write(str(m_p[i])+"  "+str(m_v[i])+"  "+str(ideal_p[i])+"  "+str(ideal_v[i])+'\n')
    test_const_vel()

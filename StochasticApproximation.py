import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
def Robbins_Monro():
    total_steps = 50
    w0 = 0.0
    ita_list = []
    N0 = 50
    w0_list = []
    for k in range(1,total_steps):
        ita = random.gauss(0, 1)
        ita_list.append(ita)
        w0_list.append(w0)
        func = pow(w0,3) - 5 + ita
        ak = 1 / (N0 + k)
        w1 = w0 - ak * func
        w0 = w1

    x = range(1,total_steps)

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)  # 2行2列，第1个子图
    plt.plot(x, ita_list)
    plt.title('ita')

    plt.subplot(2, 1, 2)  # 第2个子图
    plt.plot(x, w0_list)
    plt.title('w*')

    plt.tight_layout()    # 自动调整子图间距
    plt.show()
def Sampling(N = 100):
    mean = 0
    std = 20
    lower, upper = -20, 20

    a = (lower - mean) / std
    b = (upper - mean) / std

    x_samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=N)
    y_samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=N)
    # sampels = [(x_samples[i],y_samples[i]) for i in range(len(x_samples))]
    return x_samples,y_samples
def StochasticApproxiamtion(): 
    total_samples = 100
    total_iteration = 30
    x_samples,y_samples = Sampling(total_samples)
    w0 = (-20,20)
    sgd_list = [w0]
    bgd_list = [w0]
    mbgd_list = [w0]
    sgd_distance_list = [math.sqrt(w0[0]**2 + w0[1]**2)]
    bgd_distance_list = [math.sqrt(w0[0]**2 + w0[1]**2)]
    mbgd_distance_list = [math.sqrt(w0[0]**2 + w0[1]**2)]
    # SGD
    for i in range(1,total_iteration):
        ak = 1 / (i + 10)
        item = random.randint(0,total_samples -1)
        pk = (x_samples[item],y_samples[item])
        wk = (w0[0] - ak * (w0[0] - pk[0]),w0[1] - ak * (w0[1] - pk[1]))
        w0 = wk
        sgd_list.append(w0)
        sgd_distance_list.append(math.sqrt(w0[0]**2 + w0[1]**2))
    sgd_line_x, sgd_line_y = zip(*sgd_list)


    # BGD
    x_bar = np.mean(x_samples)
    y_bar = np.mean(y_samples)
    pk = (x_bar,y_bar)
    for i in range(1,total_iteration):
        ak = 1 / (i + 10)
        wk = (w0[0] - ak * (w0[0] - pk[0]),w0[1] - ak * (w0[1] - pk[1]))
        w0 = wk
        bgd_list.append(w0)
        bgd_distance_list.append(math.sqrt(w0[0]**2 + w0[1]**2))
    bgd_line_x, bgd_line_y = zip(*bgd_list)

    # MBGD
    m = 50
    for i in range(1,total_iteration):
        ak = 1 / (i + 10)
        xm_bar = 0.0
        ym_bar = 0.0
        for _ in range(m):
            item = random.randint(0,total_samples -1)
            xm_bar = xm_bar + x_samples[item]
            ym_bar = ym_bar + y_samples[item]
        pk = (xm_bar / m,ym_bar / m)
        wk = (w0[0] - ak * (w0[0] - pk[0]),w0[1] - ak * (w0[1] - pk[1]))
        w0 = wk
        mbgd_list.append(w0)
        mbgd_distance_list.append(math.sqrt(w0[0]**2 + w0[1]**2))
    mbgd_line_x, mbgd_line_y = zip(*mbgd_list)
    plt.figure(figsize=(18, 18))

    plt.subplot(2, 1, 1)
    plt.plot(sgd_line_x,sgd_line_y, linestyle='-', color='purple', linewidth=2,label ='sgd') 
    plt.plot(bgd_line_x,bgd_line_y, linestyle='-', color='red', linewidth=2,label ='bgd') 
    plt.plot(mbgd_line_x,mbgd_line_y, linestyle='-', color='green', linewidth=2,label ='mbgd') 
    plt.scatter(x_samples, y_samples, color='blue', alpha=0.7,marker='o',facecolors = 'None',label ='sample') 
    plt.scatter(0, 0, color='black', alpha=1.0,marker='o',facecolors = 'None',linewidths=3.0,label ='mean') 
    plt.legend()
    plt.title("散点图示例")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('scatter')

    plt.subplot(2, 1, 2)  # 第2个子图
    plt.plot(range(0,total_iteration),sgd_distance_list,marker='o',linestyle='-',label = 'SGD(m=1)')
    plt.plot(range(0,total_iteration),bgd_distance_list,marker='o',linestyle='-',label = 'BGD(m=100)')
    plt.plot(range(0,total_iteration),mbgd_distance_list,marker='o',linestyle='-',label = 'MBGD(m=50)')
    plt.title("迭代步数")
    plt.xlabel("步数")
    plt.ylabel("到真实值的距离")
    plt.tight_layout() 
    plt.legend()
    plt.show()

def main():
    # Robbins_Monro()
    StochasticApproxiamtion()
if __name__ == '__main__':
    main()
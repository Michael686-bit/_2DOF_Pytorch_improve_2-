"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
# from final.env import ArmEnv
# from final.rl import DDPG

# from _2DOF_Pytorch_test.env import ArmEnv
# # from rl import DDPG
# from _2DOF_Pytorch_test.rl_torch import DDPG

from env import ArmEnv
# from rl import DDPG
from rl_torch import DDPG

MAX_EPISODES = 1800
MAX_EP_STEPS = 300
ON_TRAIN = 0 #True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    reward_all = []
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        
        env.arm_info['r'] 
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()

            a = rl.choose_action(s)

            s_, r, done, _ = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                reward_all.append(ep_r)
                break

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    import os

    plt.figure(figsize=(10, 6))
    plt.ylabel('reward_all')
    plt.xlabel('training steps')
    plt.plot(np.arange(len(reward_all)), reward_all)
    # rl.save()  rl.save()
    # plt.show()

    # from datetime import datetime
    #
    # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = rl.save()
    file_name = f'params_{current_time}.png'  # 文件名
    save_path = './model_save'
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)

    # 存数据
    import pandas as pd

    # 创建 DataFrame
    df = pd.DataFrame({
        'len(reward_all)': len(reward_all),
        'reward_all': reward_all
    })
    file_name = f'params_{current_time}.xlsx'  # 文件名
    save_path = './model_save'
    file_path = os.path.join(save_path, file_name)
    # 保存 DataFrame 到 Excel 文件
    df.to_excel(file_path, index=False)
    print(f"save train result as {file_name}")
    plt.show()




def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    print(f"s = {s}")
    env.set_goal(240,240)
    timer = 0
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done, angle_all = env.step(a)
        print(f"angle_all = {angle_all}")

        # timer +=1
        # if timer % 800 == 200:
        #     env.set_goal(100, 300)
        # if timer % 800 == 400:
        #     env.set_goal(100, 100)
        # if timer % 800 == 600:
        #     env.set_goal(300, 100)
        # if timer % 800 == 0:
        #     env.set_goal(300, 300)


def eval_p2p():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    # s = env.reset()
    s = env.reset_start()
    print(f"s = {s}")
    # env.set_goal(200-42.5 +0, 200+39.23 +0)  #[42.5 , 39.23]
    env.set_goal(200 -42.5 , 200-39.23 )
    done = 0
    done_4p = 0
    timer = 0
    traj_all = []
    traj_q_all = []

    ang_traj = []
    while not done_4p:
        env.render()
        a = rl.choose_action(s)
        s, r, done, angle_all = env.step(a)
        print(f"s = {s}")
        traj_all.append((s[2]*200-200,s[3]*200-200))# 坐标平移转化
        traj_q_all.append((angle_all[0],angle_all[1]))
        print(f"angle_all = {angle_all}")
        timer += 1

        if timer > 200:
            done_4p = 1

            # env.set_goal(220, 220)
        # if timer % 800 == 600:
        #     env.set_goal(100, 100)
        # if timer % 800 == 0:
        #     env.set_goal(300, 100)


    x_vals = [point[0] for point in traj_all]
    y_vals = [point[1] for point in traj_all]

    q1_vals = [point[0] for point in traj_q_all]
    q2_vals = [point[1] for point in traj_q_all]
    print(f"q1_vals = {q1_vals}")
    print(f"q2_vals = {q2_vals}")


    # 保存数据
    import pandas as pd
    # 创建 DataFrame
    df = pd.DataFrame({
        'x_vals': x_vals,
        'y_vals': y_vals,
        'q1_vals': q1_vals,
        'q2_vals': q2_vals

    })


    import os
    save_path = '.\data_process\data'  

    # 确保文件夹存在
    os.makedirs(save_path, exist_ok=True)

    from datetime import datetime

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'Exy_q12_{current_time}.xlsx'  # 文件名
    file_path_all = os.path.join(save_path, file_name)  # 完整文件路径    

    # 保存 DataFrame 到 Excel 文件
    df.to_excel(file_path_all, index=False)


    # 画图
    import matplotlib.pyplot as plt

   
    # 第一部分：绘制二维曲线
    fig1 = plt.figure()  # 创建第一个图形对象
    ax1 = fig1.add_subplot(111)  # 在第一个图形对象中添加子图
    ax1.plot(x_vals, y_vals, label='Data Curve')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.legend()
    ax1.set_title('end-effector position')

    # 第二部分：绘制关节角度图像
    fig2 = plt.figure()  # 创建第二个图形对象
    ax2 = fig2.add_subplot(111)  # 在第二个图形对象中添加子图

    # 对数据进行处理
    q1_vals = [0 if x > 6.18 else x for x in q1_vals]
    q2_vals = [0 if x > 6.18 else x for x in q2_vals]

    ax2.plot(q1_vals, q2_vals, label='Joint Angles')
    ax2.set_xlabel('q1_vals')
    ax2.set_ylabel('q2_vals')
    ax2.legend()
    ax2.set_title('Joint Angles Plot')

    # 显示所有图形
    plt.show()


if ON_TRAIN:
    train()
else:
    # eval()
    eval_p2p()
    # cde = 0




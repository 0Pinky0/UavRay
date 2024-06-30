## 对手建模

参考文献：https://arxiv.org/abs/2109.06783

- cvae.py: 使用GRU作为编码器与解码器的Conditional VAE
- pretext_dataset.npy: 对手历史坐标数据，格式为归一化后的坐标
- pretext_dataset.py: 使用Pytorch中Dataset封装的数据集类
- pretext_train.py: 训练对手建模模型
- deep_q_net.py: 用于演示的DQN网络

使用时，将对手历史数据作为序列输入到CVAE的编码器中以产生隐向量，并将隐向量与环境观测一并输入决策网络；

简要示意：

（需要修改环境将对手的位置信息用info传出来）

````python
import torch
from boat_env.env import BoatEnv
from pretext.cvae import PretextVAE
from pretext.deep_q_net import DeepQNet
from collections import deque

# 初始化模型
cvae = PretextVAE()
dqn = DeepQNet()
# 初始化环境
env = BoatEnv()
obs, info = env.reset()
done = False
# 记录历史数据
opponent_history = deque(maxlen=20)
while not done:
    opponent_history.append(info['opponent_position'])
    z = cvae.encoder.predict(
        x=torch.tensor(opponent_history),
        each_seq_len=len(opponent_history),
    )
    q_val = dqn(obs, z)
    action = int(torch.argmax(q_val))
    obs, reward, done, terminated, info = env.step(action)
````
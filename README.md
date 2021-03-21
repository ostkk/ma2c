# Multi-agent A2C

1.A2C算法修改自莫烦
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py

2.多智能体环境:ma_gym中的Switch2-v1，用法与gym基本相同
https://github.com/koulanurag/ma-gym

# agent

![Switch2-v0.gif](https://raw.githubusercontent.com/koulanurag/ma-gym/master/static/gif/Switch2-v0.gif)

It's a grid world environment having n agents where each agent wants to move their corresponding home location ( marked in boxes outlined in same colors). The challenging part of the game is to pass through the narrow corridor through which only one agent can pass at a time. They need to coordinate to not block the pathway for the other. A reward of +5 is given to each agent for reaching their home cell. The episode ends when both agents has reached their home state or for a maximum of 100 steps in environment.

|    Name    |                         Description                          |
| :--------: | :----------------------------------------------------------: |
| Switch2-v0 |   Each agent receives only it's local position coordinates   |
| Switch2-v1 | Each agent receives position coordinates of all other agents |

当前使用的是v1，环境的调用方法与gym完全一致

# result

![image-20210321175417967](img\image-20210321175417967.png)



使用的架构是完全去中心化，经过测试，完全中心化和中心化训练、去中心化执行的架构方式都无法使模型到达最佳的收敛状态。
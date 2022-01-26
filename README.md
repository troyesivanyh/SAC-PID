# SAC-PID

##Note
Considering that gazebo and ros environments are difficult to configure, I rewrote a version in pybullet.


##Paper
This is the offical implementation of the paper:
[A self-adaptive SAC-PID control approach based on reinforcement learning for mobile robots](https://onlinelibrary.wiley.com/doi/abs/10.1002/rnc.5662). 

This paper was published in International Journal of Robust and Nonlinear Control (IJRNC)



##Video
- [Youtube version](https://youtu.be/GaWI_T6etUM)
- [bilibili version](https://www.bilibili.com/video/BV11q4y1c7Vn?from=search)

##Requriments
python >=3.6

torch>=1.0

pybullet

##Train
```bash
python main.py
```

##Citation
If you find our work useful for your research, please consider citing the paper:
``` 
@article{https://doi.org/10.1002/rnc.5662,
author = {Yu, Xinyi and Fan, Yuehai and Xu, Siyu and Ou, Linlin},
title = {A self-adaptive SAC-PID control approach based on reinforcement learning for mobile robots},
journal = {International Journal of Robust and Nonlinear Control},
volume = {n/a},
number = {n/a},
pages = {},
keywords = {hierarchical structure, mobile robots, reinforcement learning, SAC-PID control},
doi = {https://doi.org/10.1002/rnc.5662},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/rnc.5662},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/rnc.5662},
abstract = {Abstract Proportional–integral–derivative (PID) control is the most widely used in industrial control, robot control, and other fields. However, traditional PID control is not competent when the system cannot be accurately modeled and the operating environment is variable in real time. To tackle these problems, we propose a self-adaptive model-free SAC-PID control approach based on reinforcement learning for automatic control of mobile robots. A new hierarchical structure is developed, which includes the upper controller based on soft actor-critic (SAC), one of the most competitive continuous control algorithms, and the lower controller based on incremental PID controller. SAC receives the dynamic information of the mobile robot as input and simultaneously outputs the optimal parameters of incremental PID controllers to compensate for the error between the path and the mobile robot in real time. In addition, the combination of 24-neighborhood method and polynomial fitting is developed to improve the adaptability of SAC-PID control method to complex environment. The effectiveness of the SAC-PID control method is verified with several different difficulty paths both on Gazebo and real mecanum mobile robot. Furthermore, compared with fuzzy PID control, the SAC-PID method has merits of strong robustness, generalization, and real-time performance.}
```



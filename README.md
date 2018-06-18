# Finite_Element_Method 有限元方法
## 程序目的：
![](https://github.com/yellowyellowyao/-Finite_Element_Method/blob/master/picture/%E6%95%B0%E5%AD%A6%E9%97%AE%E9%A2%98%E6%8F%8F%E8%BF%B0.png)
本项目使用有限元方法，分别对三角形剖分和矩形剖分进行处理，其中三角形剖分借用easymesh软件剖分的网格。
本文件包含
- 包含3套不同疏密的easymesh三角形网格剖分文件夹
- easymesh生成的文件说明书:  EasyMesh 说明书.docx
- 基本版的解决方案： magic_tool.py
- 进阶版的解决方案： more_powerful_magic_tool.py

import pandas as pd
import numpy as np
from matlotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from sympy import *



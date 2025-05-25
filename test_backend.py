#!/usr/bin/env python3

import matplotlib
print(f"Initial backend: {matplotlib.get_backend()}")

# 强制设置为TkAgg
matplotlib.use('TkAgg', force=True)
print(f"After setting TkAgg: {matplotlib.get_backend()}")

import matplotlib.pyplot as plt
print(f"After importing pyplot: {matplotlib.get_backend()}")

# 测试简单的绘图
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Test Plot")

try:
    plt.show()
    print("plt.show() executed successfully")
except Exception as e:
    print(f"Error with plt.show(): {e}")

print("Test completed") 
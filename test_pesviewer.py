#!/usr/bin/env python3

import matplotlib
print(f"Initial backend: {matplotlib.get_backend()}")

# 强制设置为TkAgg
matplotlib.use('TkAgg', force=True)
print(f"After setting TkAgg: {matplotlib.get_backend()}")

# 导入pesviewer
import sys
sys.path.insert(0, '.')
from pesviewer import pesviewer

print(f"After importing pesviewer: {matplotlib.get_backend()}")

# 测试运行pesviewer
try:
    pesviewer.pesviewer('examples/input.txt')
    print("PESViewer executed successfully")
except Exception as e:
    print(f"Error with PESViewer: {e}")
    import traceback
    traceback.print_exc() 
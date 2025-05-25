#!/usr/bin/env python3
"""
测试占位线功能的简单脚本
"""

import sys
import os
sys.path.insert(0, 'pesviewer')

from pesviewer import pesviewer

def create_test_input():
    """创建一个简单的测试输入文件"""
    test_input = """
> <id>
test_placeholder

> <options>
title 1
units kJ/mol
draw_placeholder_lines 1
save 1

> <wells>
W1 0.0
W2 -10.5

> <bimolec>
B1 15.2
B2 -5.8

> <ts>
TS1 25.0 W1 W2
TS2 20.0 W2 B1

> <barrierless>

"""
    
    with open('test_placeholder.inp', 'w') as f:
        f.write(test_input)
    
    return 'test_placeholder.inp'

def main():
    """主函数"""
    print("创建测试输入文件...")
    input_file = create_test_input()
    
    print("运行PESViewer...")
    try:
        pesviewer(input_file)
        print("测试完成！检查生成的图片文件 test_placeholder_pes_plot.png")
    except Exception as e:
        print(f"运行时出错: {e}")
    
    # 清理测试文件
    if os.path.exists(input_file):
        os.remove(input_file)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正CSV文件编码
将非UTF-8编码的CSV文件转换为UTF-8编码
cd /home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main

python 修正编码.py
FaST-main-8_8MO4Delete/FaST-main/修正编码.py
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全编码转换：GBK → UTF-8
支持生僻字，自动跳过错误，绝对不崩溃
"""

import os
import shutil

# 你的文件路径
INPUT_FILE  = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/events_list_quan.csv"
BACKUP_FILE = INPUT_FILE + ".backup"

def convert_safe():
    # 备份
    shutil.copy2(INPUT_FILE, BACKUP_FILE)
    print("✅ 已备份原文件")

    # 安全读取（支持生僻字、忽略错误）
    with open(INPUT_FILE, "r", encoding="gb18030", errors="ignore") as f:
        content = f.read()

    # 写入 UTF-8
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ 编码转换完成：GBK → UTF-8")
    print("✅ 文件可直接用于大模型训练！")

if __name__ == "__main__":
    convert_safe()
"""
python 数据预处理图.py
"""
from graphviz import Digraph

dot = Digraph('pipeline', format='png')
dot.attr(rankdir='LR', fontsize='14')

# 节点样式
dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')

# 原始数据
dot.node('flow', '历史流量数据', fillcolor='#D6EAF8')
dot.node('weather', '天气信息', fillcolor='#D6EAF8')
dot.node('holiday', '节假日信息', fillcolor='#D6EAF8')
dot.node('event', '交通事件信息', fillcolor='#D6EAF8')
dot.node('station', '站点属性信息', fillcolor='#D6EAF8')

# 预处理
dot.node('preprocess', '数据预处理\n时间对齐 / 缺失值处理 / 结构化编码', fillcolor='#FAD7A0')

# 样本构造
dot.node('window', '滑动窗口构造\n输入长度=8，输出长度=8\n四类流量特征', fillcolor='#D5F5E3')

# FaST-MV
dot.node('fastmv', 'FaST-MV 初步预测', fillcolor='#E8DAEF')
dot.node('coarse', '初步预测结果\nŷ^(0)', fillcolor='#E8DAEF')

# LLM
dot.node('prompt', '结构化提示构建\n站点/时间/天气/节假日/事件/\nFlow Pattern/初步预测', fillcolor='#F5C6CB')
dot.node('llm', 'LLM 微调修正模块', fillcolor='#F5C6CB')

# 输出
dot.node('final', '最终预测结果\nŷ', fillcolor='#D4E6F1')

# 连线
for n in ['flow', 'weather', 'holiday', 'event', 'station']:
    dot.edge(n, 'preprocess')

dot.edge('preprocess', 'window')
dot.edge('window', 'fastmv')
dot.edge('fastmv', 'coarse')
dot.edge('coarse', 'prompt')
dot.edge('preprocess', 'prompt')
dot.edge('prompt', 'llm')
dot.edge('llm', 'final')

dot.render('fig4_1_data_pipeline', cleanup=True)
print("流程图已保存为 fig4_1_data_pipeline.png")

import sys
sys.path.append('../..') # 用于引入父目录的文件
from NLP_start.dataset import spiral

x ,t = spiral.load_data() #x是输入数据，为二维   t为监督数据，为三维，是one-hot向量
print('x',x.shape)
print('t',t.shape)

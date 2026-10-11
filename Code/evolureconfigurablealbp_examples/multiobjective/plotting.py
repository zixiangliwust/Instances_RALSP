import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==================== 读取数据 ====================
file_path = "D:\\GitHubCreated\\IISO_TeamResearch\\3-成员科研管理\\A0-导师科研管理\\小论文3-Reconfigurable assembly line\\Data for Figure 5.xlsx"   # 请修改为实际路径
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 分离两种方法的数据
qmohh_df = df[df["Method"] == "QMOHH"]
model_df = df[df["Method"] == "Model"]

# 提取坐标
x_q, y_q, z_q = qmohh_df["Obj1"].values, qmohh_df["Obj2"].values, qmohh_df["Obj3"].values
x_m, y_m, z_m = model_df["Obj1"].values, model_df["Obj2"].values, model_df["Obj3"].values

# ==================== 抽样简化 QMOHH 的点 ====================
sample_size = min(30, len(x_q))          # 最多显示30个点，避免拥挤
idx = np.random.choice(len(x_q), size=sample_size, replace=False)
x_q_sampled = x_q[idx]
y_q_sampled = y_q[idx]
z_q_sampled = z_q[idx]

# ==================== 绘图 ====================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 Model 解（全部，红色圆圈，半透明）
ax.scatter(x_m, y_m, z_m,
           c='red', marker='o', alpha=0.6, s=40,
           label='Model')

# 绘制 QMOHH 解（抽样，蓝色三角形，半透明）
# ax.scatter(x_q_sampled, y_q_sampled, z_q_sampled,
#            c='blue', marker='^', alpha=0.7, s=50,
#            label=f'QMOHH')

ax.scatter(x_q, y_q, z_q,
           c='blue', marker='^', alpha=0.7, s=50,
           label=f'QMOHH')

# 设置标签和视角
ax.set_xlabel('Reconfiguration cost', fontsize=12)
ax.set_ylabel('Violations of part-frequency', fontsize=12)
ax.set_zlabel('Workload equalization', fontsize=12)  # 旋转90度使文字正向
ax.view_init(elev=25, azim=-120)   # 调整视角使前沿更清晰

# 找到所有目标的最小值和最大值
min_x = min(np.min(x_m), np.min(x_q))
min_y = min(np.min(y_m), np.min(y_q))
min_z = min(np.min(z_m), np.min(z_q))
max_x = max(np.max(x_m), np.max(x_q))
max_y = max(np.max(y_m), np.max(y_q))
max_z = max(np.max(z_m), np.max(z_q))

# 在理想点位置添加一个特殊标记（伞柄底部）
ax.scatter([min_x], [min_y], [min_z], 
           c='green', marker='*', s=200, 
           label='Ideal point (min)', alpha=0.8)

# 添加图例
ax.legend(loc='upper left', fontsize=10)

# 保存图片（高分辨率）- 移除bbox_inches增大图片范围
plt.savefig('figure5.png', dpi=600)
plt.show()

# ==================== 可选：使用 Plotly 绘制交互式图形 ====================
# import plotly.express as px
# import plotly.graph_objects as go
#
# fig = go.Figure()
# fig.add_trace(go.Scatter3d(x=x_m, y=y_m, z=z_m,
#                            mode='markers',
#                            marker=dict(size=4, color='red', opacity=0.6),
#                            name='Developed model'))
# fig.add_trace(go.Scatter3d(x=x_q, y=y_q, z=z_q,
#                            mode='markers',
#                            marker=dict(size=4, color='blue', opacity=0.7),
#                            name='QMOHH'))
# fig.update_layout(scene=dict(xaxis_title='Reconfiguration cost',
#                              yaxis_title='Violations of part-frequency',
#                              zaxis_title='Workload equalization'),
#                   legend=dict(x=0.8, y=0.9))
# fig.write_html('figure5_interactive.html')
# fig.show()
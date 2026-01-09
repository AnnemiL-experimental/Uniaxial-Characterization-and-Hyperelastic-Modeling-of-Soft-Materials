import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ====== 样式设置 ======
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.labelsize": 12,
    "figure.dpi": 300,
})

# ====== 读取数据函数 ======
def load_clean(filename):
    if not os.path.exists(filename): return None
    try:
        df = pd.read_csv(filename, skiprows=1, header=None, dtype=str)
        for c in df.columns: df[c] = df[c].str.replace('"', '').str.strip()
        return df
    except: return None

# ====== 1. 准备 Tension 数据 (Cauchy) ======
# 使用之前拟合好的文件 (ogden_fit_strip_1.csv)
# 里面已经是 Cauchy Stress 了
df_tensile = pd.read_csv("ogden_fit_strip_1.csv")
ten_lam = df_tensile['lambda']
ten_stress = df_tensile['sigma_c_data_MPa'] # Cauchy

# ====== 2. 准备 Compression 数据 (转换为 Cauchy) ======
# 读取 Silicone Sample 1 (Raw Data)
comp_file = "Li_silicone sample 1.csv"
dia = 18.8
h = 23.4

df_comp = load_clean(comp_file)
# 解析: 0=Time, 1=Load(N), 4=Position(mm) (假设结构)
# 实际上我们要重新解析一下
load_N = pd.to_numeric(df_comp.iloc[:, 1], errors='coerce')
# 如果没有 Position 列，用 load 估算? 不，你的文件里应该有 Position
# 让我们尝试找 Position 列
if df_comp.shape[1] > 4:
    pos_mm = pd.to_numeric(df_comp.iloc[:, 4], errors='coerce')
else:
    # 找不到就跳过
    pos_mm = pd.Series(np.zeros(len(load_N)))

# 清洗
valid = ~np.isnan(load_N) & ~np.isnan(pos_mm)
load_N = load_N[valid]
pos_mm = pos_mm[valid]

# 截取 Loading Phase (Load 增加)
idx_peak = load_N.idxmax()
load_N = load_N.iloc[:idx_peak]
pos_mm = pos_mm.iloc[:idx_peak]

# 计算
area = np.pi * (dia/2)**2
# Engineering Stress (Negative for compression)
eng_stress = -1 * (load_N / area) 
# Stretch
lam = 1.0 - (pos_mm / h)

# *** 关键：转换为 Cauchy Stress 以匹配 Tension ***
# Cauchy = Engineering * Lambda
comp_cauchy_stress = eng_stress * lam

# ====== 3. 绘图 ======
fig, ax = plt.subplots(figsize=(8, 6))

# 画 Tension (Red)
ax.plot(ten_lam, ten_stress, color="#8B0000", linewidth=2, label="Tension")

# 画 Compression (Blue)
ax.plot(lam, comp_cauchy_stress, color="#1f77b4", linewidth=2, label="Compression")

# 画辅助线 (原点)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
ax.axvline(1, color='black', linewidth=0.8, linestyle='-')

# 装饰
ax.set_xlabel(r"Stretch Ratio, $\lambda$ (-)")
ax.set_ylabel(r"Cauchy Stress, $\sigma$ (MPa)")
ax.set_title(r"Full Range Mechanical Response of Silicone")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.5)

# 标注区域
ax.text(1.5, 0.2, "Tension\n($\lambda > 1$)", ha='center', color='#8B0000')
ax.text(0.9, -0.2, "Compression\n($\lambda < 1$)", ha='center', color='#1f77b4')

plt.tight_layout()
plt.savefig("Figure12_Full_Response.png")
print("✅ 对比图已生成: Figure12_Full_Response.png")
plt.show()
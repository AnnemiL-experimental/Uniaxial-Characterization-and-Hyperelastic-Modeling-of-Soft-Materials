import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ====== 1. 样式设置  ======
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

# ====== 2. 样本尺寸  ======
# Silicone Cylinder Specimen Dimensions
specs = [
    {"file": "Li_silicone sample 1.csv", "dia": 18.8, "h": 23.4, "label": "Sample 1"},
    {"file": "Li_silicone sample 2.csv", "dia": 18.1, "h": 22.8, "label": "Sample 2"},
    {"file": "Li_silicone sample 3.csv", "dia": 18.5, "h": 23.1, "label": "Sample 3"}
]

# ====== 3. 定义 Ogden N=1 模型 (Engineering Stress) ======
def ogden_n1_stress(lam, mu, alpha):
    """
    Ogden N=1 Uniaxial Stress Formula (Incompressible)
    Input: lam (stretch ratio)
    Output: Engineering Stress (MPa) - Negative for compression
    """
    # 避免除以0
    if abs(alpha) < 1e-9: alpha = 1e-9
    
    # 公式: sigma = (2*mu/alpha) * (lambda^(alpha-1) - lambda^(-alpha/2 - 1))
    return (2 * mu / alpha) * (lam**(alpha - 1) - lam**(-alpha / 2 - 1))

# ====== 4. 数据读取与处理 ======
all_lam = []
all_stress = []  # Store negative stress for fitting physics
plot_data = []   # Store data for plotting

print(">>> 正在处理压缩数据...")

for s in specs:
    if not os.path.exists(s["file"]):
        print(f"❌ 找不到文件: {s['file']}")
        continue
    
    try:
        # 读取 CSV (跳过可能的表头单位行)
        # 文件里有 "Time (sec), Load (N)..."
        df = pd.read_csv(s["file"], skiprows=1) 
        
        # 清洗列名 (去空格)
        df.columns = [c.strip() for c in df.columns]
        
        # 确保是数字
        df['Load (N)'] = pd.to_numeric(df['Load (N)'], errors='coerce')
        df['Position (mm)'] = pd.to_numeric(df['Position (mm)'], errors='coerce')
        df.dropna(subset=['Load (N)', 'Position (mm)'], inplace=True)
        
        # --- 截取 Loading Phase ---
        # 压缩是 Load 增加的过程，找到最大 Load 的点
        idx_peak = df['Load (N)'].idxmax()
        loading = df.iloc[:idx_peak+1].copy()
        
        # 过滤掉初始接触前的噪音 (Load < 0.05 N)
        loading = loading[loading['Load (N)'] > 0.05]
        
        # --- 计算物理量 ---
        # Area (mm^2)
        area = np.pi * (s['dia'] / 2)**2
        
        # Engineering Stress (MPa) = Force / Original Area
        # 物理上压缩力对应负应力，用于拟合
        stress_neg = -1 * (loading['Load (N)'] / area)
        
        # Stretch Lambda = 1 - (Delta_L / L0)
        # Position 是压缩位移 (正值)
        lam = 1.0 - (loading['Position (mm)'] / s['h'])
        
        # 收集用于拟合的数据 (Negative Stress, Lambda < 1)
        all_lam.append(lam)
        all_stress.append(stress_neg)
        
        # 收集用于画图的数据 (Absolute values for 1st Quadrant plot)
        plot_data.append({
            "lam": lam,
            "stress_abs": -stress_neg, # 正值用于画图
            "label": s["label"]
        })
        
        print(f"✅ {s['label']}: Max Strain = {(1-lam.min())*100:.1f}%, Max Stress = {-stress_neg.min():.3f} MPa")

    except Exception as e:
        print(f"❌ 处理 {s['file']} 出错: {e}")

# ====== 5. 执行拟合 (Fit against Negative Stress) ======
if all_lam:
    X_fit = pd.concat(all_lam)
    y_fit = pd.concat(all_stress)
    
    # 初始猜测: mu=0.5 MPa (类似橡胶), alpha=2.0 (类似 Neo-Hookean)
    p0 = [0.5, 2.0]
    # 约束: mu 必须 > 0
    bounds = ([0, -np.inf], [np.inf, np.inf])
    
    try:
        popt, pcov = curve_fit(ogden_n1_stress, X_fit, y_fit, p0=p0, bounds=bounds)
        mu_fit, alpha_fit = popt
        
        # 计算 SSE / R2
        y_pred = ogden_n1_stress(X_fit, *popt)
        sse = np.sum((y_fit - y_pred)**2)
        sst = np.sum((y_fit - np.mean(y_fit))**2)
        r2 = 1 - (sse/sst)
        
        print("\n" + "="*40)
        print(">>> 压缩拟合结果 (Ogden N=1) <<<")
        print(f"mu (Shear Modulus) = {mu_fit:.4f} MPa")
        print(f"alpha (Exponent)   = {alpha_fit:.4f}")
        print(f"SSE                = {sse:.5f}")
        print(f"R-squared          = {r2:.5f}")
        print("="*40)
        
        # ====== 6. 画图 (第一象限: Absolute Stress vs Stretch) ======
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # A. 画实验数据
        styles = ['-', '--', ':']
        colors = ['black', 'dimgray', 'silver']
        
        min_lam_plot = 1.0
        
        for i, d in enumerate(plot_data):
            ax.plot(d['lam'], d['stress_abs'], 
                    color=colors[i], linestyle=styles[i], linewidth=1.5,
                    label=f"Exp: {d['label']}")
            if d['lam'].min() < min_lam_plot:
                min_lam_plot = d['lam'].min()
        
        # B. 画拟合曲线
        # 生成 Lambda 序列 (从 1.0 降到 最小Lambda)
        lam_smooth = np.linspace(1.0, min_lam_plot, 100)
        # 计算理论负应力
        stress_pred_neg = ogden_n1_stress(lam_smooth, mu_fit, alpha_fit)
        # 取绝对值画图
        ax.plot(lam_smooth, -stress_pred_neg, 
                color="#8B0000", linestyle="-", linewidth=2.5,
                label=f"Ogden N=1 Fit")
        
        # C. 信息框
        textstr = '\n'.join((
            r'Compression Fit (N=1):',
            rf'$\mu = {mu_fit:.3f}$ MPa',
            rf'$\alpha = {alpha_fit:.3f}$',
            rf'$R^2 = {r2:.3f}$'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
        ax.text(0.02, 0.18, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # D. 装饰
        ax.set_xlabel(r"Stretch Ratio, $\lambda$ (-)")
        ax.set_ylabel(r"Engineering Stress, $P$ (MPa)") # 标明是绝对值或 Compressive
        ax.set_title(r"Silicone Uniaxial Compression Response")
        
        # 坐标轴反转 X轴? 
        # 通常 Compression Stretch 画图是从 1.0 向左减小，或者从右向左。
        # 这里我们就保持常规坐标轴: 1.0 在右边，0.8 在左边
        ax.set_xlim(1.005, min_lam_plot * 0.98) # X轴逆序显示更直观? 不，保持数学顺序
        ax.invert_xaxis() # <--- 让 X 轴从 1.0 (左) 到 0.8 (右) 也就是 Loading 方向
        
        ax.set_ylim(bottom=0)
        ax.minorticks_on()
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc="upper right", frameon=False) # 改到左上，避开曲线
        
        plt.tight_layout()
        plt.savefig("Compression_Ogden_N1.png", dpi=300)
        print("✅ 图表已生成: Compression_Ogden_N1.png")
        plt.show()

    except Exception as e:
        print(f"拟合计算失败: {e}")
else:
    print("没有有效数据用于拟合。")

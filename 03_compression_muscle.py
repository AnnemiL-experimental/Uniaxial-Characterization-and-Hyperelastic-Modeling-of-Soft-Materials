import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ====== 1. 样式设置 ======
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "axes.linewidth": 1.0,
})

# ====== 2. 样本尺寸  ======
# Porcine Muscle Dimensions
specs = [
    {"file": "Li_muscle sample 1.csv", "dia": 14.0, "h": 21.0, "label": "Muscle #1"},
    {"file": "Li_muscle sample 2.csv", "dia": 14.0, "h": 22.0, "label": "Muscle #2"},
    {"file": "Li_muscle sample 3.csv", "dia": 15.0, "h": 15.0, "label": "Muscle #3"}
]

# ====== 3. Ogden N=1 模型 (输出单位调整为 kPa) ======
def ogden_n1_stress_kPa(lam, mu_kPa, alpha):
    """
    Ogden N=1 Stress (kPa).
    mu_kPa: 剪切模量 (kPa)
    """
    if abs(alpha) < 1e-9: alpha = 1e-9
    # 计算工程应力 (Compressive stress is negative physically)
    sigma = (2 * mu_kPa / alpha) * (lam**(alpha - 1) - lam**(-alpha / 2 - 1))
    return sigma

# ====== 4. 数据处理 ======
all_lam = []
all_stress = []
plot_data = []

print(">>> 正在处理 Muscle 数据...")

for s in specs:
    if not os.path.exists(s["file"]):
        print(f"❌ 找不到文件: {s['file']}")
        continue
    
    try:
        # 读取 CSV
        df = pd.read_csv(s["file"], skiprows=1)
        # 清洗列名
        df.columns = [c.strip() for c in df.columns]
        
        # 寻找 Load 和 Position 列
        load_col = [c for c in df.columns if "load" in c.lower()][0]
        pos_col = [c for c in df.columns if "position" in c.lower()][0]

        df[load_col] = pd.to_numeric(df[load_col], errors='coerce')
        df[pos_col] = pd.to_numeric(df[pos_col], errors='coerce')
        df.dropna(subset=[load_col, pos_col], inplace=True)
        
        # --- 截取 Loading Phase ---
        # 找到最大载荷点
        idx_peak = df[load_col].idxmax()
        loading = df.iloc[:idx_peak+1].copy()
        
        # 去除预载前的噪音 (Load > 0.02 N)
        loading = loading[loading[load_col] > 0.02]
        
        if loading.empty:
            print(f"⚠️ {s['label']} 数据过滤后为空")
            continue

        # --- 计算 Stress / Stretch ---
        area = np.pi * (s['dia'] / 2)**2
        
        # Stress (kPa) = (Force(N) / Area(mm^2)) * 1000
        # 同样，为了拟合物理方程，保留负号
        stress_neg_kPa = -1 * (loading[load_col] / area) * 1000
        
        # Stretch = 1 - (Pos / H)
        lam = 1.0 - (loading[pos_col] / s['h'])
        
        # 收集数据
        all_lam.append(lam)
        all_stress.append(stress_neg_kPa)
        
        plot_data.append({
            "lam": lam,
            "stress_abs": -stress_neg_kPa, # 画图用绝对值
            "label": s["label"]
        })
        print(f"✅ {s['label']}: Max Stress = {-stress_neg_kPa.min():.2f} kPa")

    except Exception as e:
        print(f"❌ 处理 {s['file']} 失败: {e}")

# ====== 5. 拟合与绘图 ======
if all_lam:
    # 合并数据进行 Global Fit
    X_fit = pd.concat(all_lam)
    y_fit = pd.concat(all_stress)
    
    # 初始猜测 (Muscle通常非常软，非线性强)
    # mu 猜测 1-10 kPa, alpha 猜测 2-10
    p0 = [5.0, 5.0] 
    bounds = ([0, -np.inf], [np.inf, np.inf])
    
    try:
        popt, pcov = curve_fit(ogden_n1_stress_kPa, X_fit, y_fit, p0=p0, bounds=bounds)
        mu_fit, alpha_fit = popt
        
        # 计算 R2
        y_pred = ogden_n1_stress_kPa(X_fit, *popt)
        sse = np.sum((y_fit - y_pred)**2)
        sst = np.sum((y_fit - np.mean(y_fit))**2)
        r2 = 1 - (sse/sst)
        
        print("\n" + "="*40)
        print(f"Muscle Fit Results (kPa):")
        print(f"mu = {mu_fit:.4f} kPa")
        print(f"alpha = {alpha_fit:.4f}")
        print(f"R2 = {r2:.4f}")
        print("="*40)
        
        # --- 画图 ---
        fig, ax = plt.subplots(figsize=(8, 5))
        
        styles = ['-', '--', ':']
        colors = ['black', 'dimgray', 'silver']
        min_lam_val = 1.0
        
        # A. 画实验曲线
        for i, d in enumerate(plot_data):
            ax.plot(d['lam'], d['stress_abs'], 
                    color=colors[i], linestyle=styles[i], linewidth=1.5,
                    label=f"Exp: {d['label']}")
            if d['lam'].min() < min_lam_val:
                min_lam_val = d['lam'].min()
        
        # B. 画拟合曲线
        lam_smooth = np.linspace(1.0, min_lam_val, 100)
        stress_pred = ogden_n1_stress_kPa(lam_smooth, mu_fit, alpha_fit)
        
        ax.plot(lam_smooth, -stress_pred, 
                color="#8B0000", linestyle="-", linewidth=2.5,
                label="Ogden N=1 Fit")
        
        # C. 参数框
        textstr = '\n'.join((
            r'Muscle Fit (N=1):',
            rf'$\mu = {mu_fit:.2f}$ kPa',
            rf'$\alpha = {alpha_fit:.2f}$',
            rf'$R^2 = {r2:.3f}$'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
        # 放在左上角或合适位置
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # D. 装饰
        ax.set_xlabel(r"Stretch Ratio, $\lambda$ (-)")
        ax.set_ylabel(r"Engineering Stress, $P$ (kPa)") # 注意单位是 kPa
        ax.set_title(r"Porcine Muscle Compression Response")
        
        # 坐标轴
        ax.set_xlim(1.005, min_lam_val * 0.98)
        ax.invert_xaxis() # 反转X轴
        ax.set_ylim(bottom=0)
        
        ax.minorticks_on()
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(True, linestyle=':', alpha=0.4)
        
        ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor='black')
        
        save_name = "Muscle_Compression_Fit.png"
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"✅ 图表已生成: {save_name}")
        plt.show()

    except Exception as e:
        print(f"拟合计算错误: {e}")
else:
    print("没有数据。")
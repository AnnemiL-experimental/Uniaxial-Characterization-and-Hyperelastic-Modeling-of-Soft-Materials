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

# ====== 1. 强力读取函数 ======
def load_data_clean(filename):
    if not os.path.exists(filename):
        print(f"❌ 找不到文件: {filename}")
        return None
    try:
        # 读取所有内容为字符串
        df = pd.read_csv(filename, skiprows=1, header=None, dtype=str)
        # 清洗引号
        for c in df.columns:
            df[c] = df[c].str.replace('"', '').str.strip()
        return df
    except Exception as e:
        print(f"读取失败 {filename}: {e}")
        return None

# ====== 2. 截取“保持阶段” (Isolate Hold Phase) ======
def get_hold_phase(time, stress, displacement, tolerance=0.02):
    """
    自动截取位移保持不变的阶段。
    tolerance: 容许的位移波动范围 (相对比例)
    """
    # 1. 找到位移最大的位置 (保持开始)
    idx_peak = displacement.idxmax()
    max_disp = displacement.max()
    
    # 2. 从峰值开始向后找，直到位移下降超过阈值 (保持结束)
    #    判断标准：位移 < 98% 的最大位移
    #    (或者直到文件结束)
    end_idx = len(displacement) - 1
    
    for i in range(idx_peak, len(displacement)):
        if displacement.iloc[i] < max_disp * (1 - tolerance):
            end_idx = i
            break
            
    # 如果保持时间太短（比如小于2秒），可能根本没有松弛阶段，直接卸载了
    if (time.iloc[end_idx] - time.iloc[idx_peak]) < 2.0:
        print("  ⚠️ 警告: 保持时间极短，可能包含卸载过程，或实验未做保持。")
    
    # 截取数据
    time_hold = time.iloc[idx_peak:end_idx].values
    stress_hold = stress.iloc[idx_peak:end_idx].values
    
    # 时间归零 (t = 0 表示松弛开始)
    time_hold = time_hold - time_hold[0]
    
    return time_hold, stress_hold, idx_peak, end_idx

# ====== 3. 通用绘图函数 ======
def plot_relaxation(title, specs, y_unit="MPa", save_name="Fig.png", is_tensile=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = ['-', '--', ':']
    colors = ['black', 'dimgray', 'gray']
    
    avg_relax_pct = []

    print(f"\n--- Processing {title} ---")

    for i, s in enumerate(specs):
        df = load_data_clean(s["file"])
        if df is None: continue
        
        try:
            # 解析数据列 
            if is_tensile:
                # Tensile: 0=Time, 1=Disp, 2=Force
                t = pd.to_numeric(df.iloc[1:, 0], errors='coerce')
                d = pd.to_numeric(df.iloc[1:, 1], errors='coerce') # Disp
                f = pd.to_numeric(df.iloc[1:, 2], errors='coerce') # Force
                
                area = s["w"] * s["t"]
                sigma = (f * 1000) / area # MPa
            else:
                # Compression: 0=Time, 1=Load, ..., 4=Position
                t = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                # 注意：有些文件可能列顺序不同，这里尝试找Position列
                # 简单起见，假设 0=Time, 1=Load, 4=Position (根据Muscle CSV)
                load_N = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                # 假设 Position 是第4列 (index 4)
                if df.shape[1] > 4:
                    d = pd.to_numeric(df.iloc[:, 4], errors='coerce')
                else:
                    # 如果找不到 Position，就假定峰值力之后就是保持
                    d = load_N # 用力来代替位移寻找峰值(通常同步)
                
                area = np.pi * (s["dia"]/2)**2
                if y_unit == "MPa":
                    sigma = load_N / area
                else:
                    sigma = (load_N / area) * 1000 # kPa

            # 去除无效值
            valid = ~np.isnan(t) & ~np.isnan(sigma) & ~np.isnan(d)
            t = t[valid].reset_index(drop=True)
            sigma = sigma[valid].reset_index(drop=True)
            d = d[valid].reset_index(drop=True)

            # === 只取保持阶段 ===
            t_plot, s_plot, idx_start, idx_end = get_hold_phase(t, sigma, d)
            
            # 只有当有足够数据时才画图
            if len(t_plot) > 10:
                ax.plot(t_plot, s_plot, color=colors[i], linestyle=styles[i], label=s["label"])
                
                # 计算松弛率
                start_stress = s_plot[0]
                end_stress = s_plot[-1]
                relax = (start_stress - end_stress) / start_stress * 100
                avg_relax_pct.append(relax)
                print(f"  {s['label']}: Duration={t_plot[-1]:.1f}s, Relax={relax:.1f}%")
            else:
                print(f"  {s['label']}: 数据段太短，无法绘制松弛图。")

        except Exception as e:
            print(f"处理 {s['file']} 出错: {e}")

    # 计算平均值并打印 
    if avg_relax_pct:
        print(f"  >>> {title} Average Relaxation: {np.mean(avg_relax_pct):.1f}%")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Stress ({y_unit})")
    ax.set_title(f"{title} Stress Relaxation")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 确保从0开始
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"✅ 图表已保存: {save_name}")

# ====== 主执行区 ======

# 1. Tensile Relaxation
specs_tensile = [
    {"file": "group 3_2_1.csv", "w": 24.75, "t": 3.12, "label": "Strip 1"},
    {"file": "group 3_2_2.csv", "w": 24.67, "t": 3.13, "label": "Strip 2"},
    {"file": "group 3_2_3.csv", "w": 24.51, "t": 3.12, "label": "Strip 3"}
]
plot_relaxation("Silicone Tensile", specs_tensile, "MPa", "Figure9_Tensile_Relax_Fixed.png", is_tensile=True)

# 2. Silicone Compression Relaxation
specs_sil_comp = [
    {"file": "Li_silicone sample 1.csv", "dia": 18.8, "label": "Sample 1"},
    {"file": "Li_silicone sample 2.csv", "dia": 18.1, "label": "Sample 2"},
    {"file": "Li_silicone sample 3.csv", "dia": 18.5, "label": "Sample 3"}
]
plot_relaxation("Silicone Compression", specs_sil_comp, "MPa", "Figure10_Silicone_Comp_Relax_Fixed.png")

# 3. Muscle Compression Relaxation
specs_mus_comp = [
    {"file": "Li_muscle sample 1.csv", "dia": 14.0, "label": "Muscle 1"},
    {"file": "Li_muscle sample 2.csv", "dia": 14.0, "label": "Muscle 2"},
    {"file": "Li_muscle sample 3.csv", "dia": 15.0, "label": "Muscle 3"}
]
plot_relaxation("Porcine Muscle Compression", specs_mus_comp, "kPa", "Figure11_Muscle_Comp_Relax_Fixed.png")
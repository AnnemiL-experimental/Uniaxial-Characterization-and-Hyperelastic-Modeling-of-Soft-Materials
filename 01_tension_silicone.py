#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze uniaxial tensile tests of three silicone strips.
- Load raw force–displacement data
- Convert to PK1 (engineering) stress and stretch
- Fit an incompressible 4-term Ogden model (N=4)
- Generate stress–stretch plots for the report
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os


# ----------------- 路径 & 尺寸 -----------------

BASE = ""   # 确认路径

tensile_files = [
    os.path.join(BASE, "group 3_2_1.csv"),
    os.path.join(BASE, "group 3_2_2.csv"),
    os.path.join(BASE, "group 3_2_3.csv"),
]

# Strip 尺寸 (mm)
strip_widths_mm  = np.array([24.75, 24.67, 24.51])
strip_thicks_mm  = np.array([3.12,  3.13,  3.12 ])
gauge_length_mm  = 200.0


# ----------------- 读 CSV & 工程应力应变 -----------------

def read_tensile_csv(path):
    """
    读取单个 tensile CSV.
    第一行是列名，第二行是单位，跳过单位那一行。
    """
    # 关键：skiprows=[1] 把 "(s),(mm),(kN)" 那行丢掉
    df = pd.read_csv(path, skiprows=[1])

    # 自动找列名，兼容带单位的写法
    time_col = [c for c in df.columns if "Time" in c][0]
    disp_col = [c for c in df.columns if "Displacement" in c][0]
    force_col = [c for c in df.columns if "Force" in c][0]

    # 转成 float，遇到奇怪字符就变 NaN
    time = pd.to_numeric(df[time_col],  errors="coerce").to_numpy()       # s
    disp_mm = pd.to_numeric(df[disp_col], errors="coerce").to_numpy()     # mm
    force_kN = pd.to_numeric(df[force_col], errors="coerce").to_numpy()   # kN

    # 去掉 NaN 行
    mask = np.isfinite(time) & np.isfinite(disp_mm) & np.isfinite(force_kN)
    time = time[mask]
    disp_mm = disp_mm[mask]
    force_kN = force_kN[mask]

    # kN → N
    force_N = force_kN * 1000.0

    return time, disp_mm, force_N


def compute_engineering_stress_strain(disp_mm, force_N, width_mm, thick_mm):
    """计算工程应力–应变."""
    eps = disp_mm / gauge_length_mm  # 工程应变

    A_mm2 = width_mm * thick_mm
    A_m2 = A_mm2 * 1e-6

    sigma_eng_Pa = force_N / A_m2
    sigma_eng_MPa = sigma_eng_Pa / 1e6
    return eps, sigma_eng_MPa


# ----------------- Ogden N=4 模型 -----------------

def ogden4_cauchy(lambda1, *params):
    """
    Ogden N=4 PK1 stress for incompressible uniaxial tension.
    params = [mu1..mu4, alpha1..alpha4] (μ 单位 MPa)
    """
    lambda1 = np.asarray(lambda1)
    mu = np.array(params[:4])
    alpha = np.array(params[4:])

    sigma = np.zeros_like(lambda1, dtype=float)
    for i in range(4):
        mui = mu[i]
        ai = alpha[i]
        # σ = 2μ/α (λ^{α-1} - λ^{-(α/2 + 1)})
        term = (2.0 * mui / ai) * (lambda1**(ai - 1.0) - lambda1**(-(ai / 2.0 + 1.0)))
        sigma += term

    return sigma   # MPa


# ----------------- 主程序 -----------------

def main():
    all_lambda = []
    all_sigma_c = []
    sample_data = []

    for i, path in enumerate(tensile_files):
        name = f"Strip_{i+1}"
        print(f"\n=== Processing {name} ===")
        print("File:", path)

        time, disp_mm, force_N = read_tensile_csv(path)

        # 只取 loading 段（位移最大之前）
        idx_max = np.argmax(disp_mm)
        time = time[:idx_max + 1]
        disp_mm = disp_mm[:idx_max + 1]
        force_N = force_N[:idx_max + 1]

        eps, sigma_eng_MPa = compute_engineering_stress_strain(
            disp_mm,
            force_N,
            strip_widths_mm[i],
            strip_thicks_mm[i],
        )

        lambda1 = 1.0 + eps
        sigma_c_MPa = sigma_eng_MPa * lambda1  # Cauchy

        all_lambda.append(lambda1)
        all_sigma_c.append(sigma_c_MPa)
        sample_data.append((name, lambda1, sigma_c_MPa))

    lam_all = np.concatenate(all_lambda)
    sig_all = np.concatenate(all_sigma_c)

    # 过滤掉明显异常点
    valid = (lam_all > 1.0) & (lam_all < 1.8) & (sig_all > 0.0)
    lam_all_fit = lam_all[valid]
    sig_all_fit = sig_all[valid]

    print("\nTotal data points used for fit:", lam_all_fit.size)

    # 初始猜测 & 边界
    p0 = [0.3, 0.1, 0.05, 0.02, 1.0, 5.0, -2.0, 0.5]
    lower = [0.0, 0.0, 0.0, 0.0, -10.0, -10.0, -10.0, -10.0]
    upper = [5.0, 5.0, 5.0, 5.0,  10.0,  10.0,  10.0,  10.0]

    print("\nFitting Ogden N=4 model ...")
    popt, pcov = curve_fit(
        ogden4_cauchy,
        lam_all_fit,
        sig_all_fit,
        p0=p0,
        bounds=(lower, upper),
        maxfev=100000,
    )

    mu_fit = popt[:4]
    alpha_fit = popt[4:]

    print("\n=== Fitted Ogden parameters (global, 3 samples) ===")
    for i in range(4):
        print(f"mu_{i+1} = {mu_fit[i]:.4f} MPa,  alpha_{i+1} = {alpha_fit[i]:.4f}")

    # 写参数到 txt
    param_txt = os.path.join(BASE, "ogden_fit_all_params.txt")
    with open(param_txt, "w", encoding="utf-8") as f:
        f.write("Global Ogden N=4 fit (3 tensile strips)\n\n")
        for i in range(4):
            f.write(f"mu_{i+1} = {mu_fit[i]:.6f} MPa\n")
        for i in range(4):
            f.write(f"alpha_{i+1} = {alpha_fit[i]:.6f}\n")
    print("\nSaved parameters to:", param_txt)

    # 为每个样品导出拟合曲线
    for (name, lambda1, sigma_c_MPa) in sample_data:
        sigma_fit = ogden4_cauchy(lambda1, *popt)
        df_out = pd.DataFrame({
            "lambda": lambda1,
            "sigma_c_data_MPa": sigma_c_MPa,
            "sigma_c_fit_MPa": sigma_fit,
        })
        out_path = os.path.join(BASE, f"ogden_fit_{name.lower()}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"Saved fit data for {name} to:", out_path)


if __name__ == "__main__":
    main()

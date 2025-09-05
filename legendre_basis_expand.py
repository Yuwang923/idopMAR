import numpy as np
import pandas as pd
from scipy.special import eval_legendre as Legendre
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
# df = pd.read_csv("D:\\idopMAR\\data\\DCM.csv")
df = pd.read_csv("D:\\idopMAR\\data\\c.csv")
# df = pd.read_csv("D:\\idopMAR\\data\\xr202408.csv")
# df = pd.read_csv("D:\\idopMAR\\data\\Norm1.csv")
df.set_index('Time', inplace=True)
# df = df.iloc[:,1:5]
# scaler_1 = StandardScaler()
# df = scaler_1.fit_transform(df)
# df = pd.DataFrame(df)

def legendre_basis_expand(df, R):
    """
    对每一列变量进行缩放到 [-1,1]，
    并计算 1..R 阶的勒让德基函数值，返回与 df 同长度的特征表。
    """
    # scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = MaxAbsScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )

    legendre_expand = pd.DataFrame(index=df.index)
    for col in df.columns:
        for r in range(1, R + 1):
            legendre_expand[f'{col}_({r})'] = [Legendre(r, y) for y in df_scaled[col]]

    return legendre_expand

legendre_basis = legendre_basis_expand(df, R=15)
legendre_basis


fig, ax = plt.subplots(figsize=(10, 6))   # 画布大小
# legendre_basis.plot(ax=ax, lw=1, marker='o',legend=False)  # 去掉图例
df.plot(ax=ax, lw=1, legend=False)
plt.tight_layout()
plt.show()

t = pd.to_numeric(legendre_basis.index, errors='raise').values.astype(float)
# t = np.linspace(0.0,10, 739)



# —— 1) 逐列“累积积分曲线” (与行数一致，首个值为0) ——
cum = cumulative_trapezoid(legendre_basis.values, x=t, axis=0, initial=0)  # 形状 (n, m)
cum_int = pd.DataFrame(
    cum, index=legendre_basis.index,
    columns=[f"{c}_int" for c in legendre_basis.columns]
)

cum_int

fig, ax = plt.subplots(figsize=(10, 6))
cum_int.plot(ax=ax, lw=1, legend=False)
ax.set_xlabel("Time")
ax.set_ylabel("Cumulative integral")
ax.set_title("Cumulative integral of Legendre features (SciPy cumulative_trapezoid)")
plt.tight_layout()
plt.show()
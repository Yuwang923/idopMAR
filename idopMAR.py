import math
import numpy as np
import pandas as pd
from scipy.special import eval_legendre as Legendre
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from sklearn.model_selection import GridSearchCV
from asgl import Regressor

# ---------------- 基础参数 ----------------
N = 105
R = 5
grid_ncols = 10                 # 每行子图数量（列数），可改
legend_on_first_only = False    # 只在第一个子图放图例，避免重复

# ---------------- 读数据 ----------------
df = pd.read_csv("D:\\idopMAR\\data\\c.csv")
df.set_index('Time', inplace=True)
df = df.iloc[:, :N]
df = pd.DataFrame(df)

# ---------------- 勒让德基展开 + 累积积分 ----------------
def legendre_basis_expand(df, R):
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

legendre_basis = legendre_basis_expand(df, R=R)
t = pd.to_numeric(legendre_basis.index, errors='raise').values.astype(float)

legendre_basis_integrate = cumulative_trapezoid(
    legendre_basis.values, x=t, axis=0, initial=0
)
legendre_basis_integrate = pd.DataFrame(
    legendre_basis_integrate, index=legendre_basis.index,
    columns=[f"{c}_int" for c in legendre_basis.columns]
)

X = legendre_basis_integrate
y_all = df

# 可选：标准化 X（若需要就把 X 替换为 X_std）
# scalerX = StandardScaler()
# X_std = pd.DataFrame(scalerX.fit_transform(X), index=X.index, columns=X.columns)

# ---------------- 组索引与权重 ----------------
group_index = [i for i in range(N) for _ in range(R)]

custom_group_weights = []
for j in range(N):
    row = [0.4] * N
    row[j] = 0.6
    custom_group_weights.append(row)

custom_individual_weights = []
for j in range(N):
    row = np.full(N * R, 0.5, dtype=float)
    row[j*R:(j+1)*R] = 0.5
    custom_individual_weights.append(row)
custom_individual_weights = np.vstack(custom_individual_weights)

# ---------------- 网格搜索参数 ----------------
param_grid = {
    'lambda1': [1e-6, 1e-3],
    'alpha': [0.5]
}

# ---------------- 画 m×n 子图 ----------------
n_targets = N
ncols = grid_ncols
nrows = math.ceil(n_targets / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.6*nrows), squeeze=False)

for i in range(n_targets):
    ax = axes[i // ncols][i % ncols]

    # 1) 为第 i 个目标构建 & 训练模型
    asgl_model_i = Regressor(
        model='lm',
        penalization='asgl',
        individual_weights=custom_individual_weights[i],
        group_weights=custom_group_weights[i],
        individual_power_weight=1,
        group_power_weight=1,
        fit_intercept=True,
        tol=1e-5
    )

    gscv_asgl = GridSearchCV(
        asgl_model_i, param_grid,
        scoring='neg_mean_squared_error', cv=10, n_jobs=-1
    )

    gscv_asgl.fit(
        X=X,                # 如果用标准化特征，则改为 X_std
        y=y_all.iloc[:, i],
        group_index=group_index
    )
    best_model = gscv_asgl.best_estimator_

    # 2) 系数、效应分解
    y_true = y_all.iloc[:, i]
    y_pred = best_model.predict(X)
    coef_series = pd.Series(best_model.coef_.ravel(), index=X.columns)

    effect = X.mul(coef_series, axis=1)
    # 合并每个变量的 R 个基的贡献 -> N 列（每列一个变量的总贡献）
    effect1 = np.array(effect).reshape(len(effect), -1, R).sum(axis=2)
    effect1 = pd.DataFrame(effect1, index=effect.index,
                           columns=[f'E{j+1}' for j in range(effect.shape[1] // R)])

    # 去掉自变量自身（第 i 列）的贡献，用于展示他变量
    effect2 = effect1.drop(columns=effect1.columns[i])
    effect2_nz = effect2.loc[:, (effect2.abs() > 1e-8).any()]

    # 3) 绘图（子图 ax）
    # 真实值
    y_true.plot(ax=ax, lw=0, marker='o', ms=4, label='y_true', color='black')
    # 其他变量贡献（细线）
    if effect2_nz.shape[1] > 0:
        effect2_nz.plot(ax=ax, lw=1, legend=False, color='green')

    # 自身贡献 + 截距（粗红线）
    indpand = effect1.iloc[:, i] + best_model.intercept_
    indpand.plot(ax=ax, lw=2.5, label='self+intercept', color='red')

    # 总和（蓝色虚线）
    row_sum = effect1.sum(axis=1) + best_model.intercept_
    row_sum.plot(ax=ax, lw=2.5, linestyle='--', label='sum(all)+intercept', color='blue')

    ax.set_title(f"Target: {y_all.columns[i]}")
    ax.grid(True, alpha=0.3)

    # 图例控制：只在第一个子图放，或都放
    if legend_on_first_only:
        if i == 0:
            ax.legend(loc='best', fontsize=9)
    else:
        ax.legend(loc='best', fontsize=9)

# 隐藏多余空白子图
total_axes = nrows * ncols
for k in range(n_targets, total_axes):
    axes[k // ncols][k % ncols].axis('off')

plt.tight_layout()
plt.show()

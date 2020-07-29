"""Gaussian Processesの概要
https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

# 2：データの生成--------------------------------
"""
SVMはスケール不変ではないため、入力データをスケールすることが必要。
入力は[0, 1]、[-1, 1]にスケーリングするか、平均0、分散1に標準化する。
"""
numSamples = 15
train_x = np.sort(10 * np.random.rand(numSamples, 1), axis=0)
_y = np.sin(train_x).ravel() * 0.6
train_y = _y + 0.02 * (np.random.randn(numSamples))
x = np.arange(0.1, 10.0, 0.1)
y = np.sin(x)
"""
[パラメーター]                [詳細]
kernel                      カーネル関数の選択。
                            デフォルトは1.0*RBF(length_scale=1.0)。
                            カーネルのハイパーパラメータはフィッティング中に最適化される。
                            カーネルの種類は以下を参照。
                            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
alpha                       カーネル関数が作る行列の逆行列計算の安定性を確保する
                            （計算時に固有値が非常に小さくなり、逆行列の計算が不安定にならないように行列の対角成分にaplhaを加える）。
optimizer                   カーネルのハイパーパラメータの最適化に使用するオプティマイザ。
                            デフォルトはscipy.optimize.minimizeの「L-BGFS-B」アルゴリズム(今現在、利用可能なものは1種類だけ)。
                            Noneを指定するとカーネルのパラメータは固定される。
n_restarts_optimizer        カーネルのハイパーパラメータを最適化する回数。0の場合は1回。
normalize_y                 予測変数の平均を0になるように正規化する。
copy_X_train                学習データをオブジェクトにコピーする。
random_state                乱数シード。

[Attributes]
X_train_
y_train_
kernel_
L_
alpha_
log_marginal_likelihood_value_
"""
kern = kernels.RBF(length_scale=1.0)
# kern = sk_kern.RationalQuadratic(length_scale=.5)
# kern = sk_kern.ConstantKernel()
# kern = sk_kern.WhiteKernel(noise_level=3.)
# kern = gp.kernels.RBF() + gp.kernels.WhiteKernel()
model = GaussianProcessRegressor(kernel=kern,  # kernel instance, default=None
                                 alpha=0.01,  # float or array-like of shape(n_sample), default=1e-10
                                 optimizer="fmin_l_bfgs_b",  # "fmin_l_bfgs_b” or callable, default="fmin_l_bfgs_b"
                                 n_restarts_optimizer=0,  # int, default=0
                                 normalize_y=False,  # boolean, optional (default: False)
                                 copy_X_train=True,  # bool, default=True
                                 random_state=None,  # int or RandomState, default=None
                                 )
_x = np.array(np.linspace(1, 9, 9))
k_samples = model.sample_y(_x.reshape(-1, 1), n_samples=5)  # カーネル関数をランダムに5つサンプリングしてくる
model.fit(train_x, train_y)
y_pred, y_std = model.predict(x.reshape(-1, 1), return_std=True)
log_marginal_likelihood = model.log_marginal_likelihood()  # 対数周辺尤度
params = model.get_params()  # 設定パラメータの取得(辞書)
scores = model.score(train_x, train_y)  # 決定係数R^2
# params = model.set_params()  # 設定パラメータの設定(辞書)

X_train = model.X_train_
y_train = model.y_train_
kernel = model.kernel_  # 予測に使用されたカーネル(最適化済みで最初に設定したパラメータとは異なる)
L = model.L_
alpha = model.alpha_
log_marginal_likelihood_value = model.log_marginal_likelihood_value_  # 対数周辺尤度

# plot
fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)
for i in range(k_samples.shape[1]):
    ax1.plot(_x, k_samples[:, i])
ax1.plot(x, y_pred, color='b', label='predict mean', zorder=2)
ax1.fill_between(x, y_pred+y_std, y_pred-y_std, facecolor='b', alpha=0.3, zorder=1)
ax1.legend()
plt.show()

# plot
fig = plt.figure(figsize=(6, 4))
ax2 = fig.add_subplot(111)
ax2.plot(x, y, color='r', label='true mean', zorder=4)
ax2.scatter(train_x, train_y, color='darkorange', label='training data', zorder=3)
ax2.plot(x, y_pred, color='b', label='predict mean', zorder=2)
ax2.fill_between(x, y_pred+y_std, y_pred-y_std, facecolor='b', alpha=0.3, zorder=1)
ax2.legend()
plt.show()

print("end")

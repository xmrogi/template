"""SVMの概要
https://scikit-learn.org/stable/modules/svm.html#shrinking-svm

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 2：データの生成--------------------------------
"""
SVMはスケール不変ではないため、入力データをスケールすることが推奨される。
入力は[0, 1]、[-1, 1]にスケーリングするか、平均0、分散1に標準化する。
"""
numSamples = 80
x = np.sort(10 * np.random.rand(numSamples, 1), axis=0)
y = np.sin(x).ravel() * 0.6
y = y + 0.1 * (np.random.randn(numSamples))
x_true = np.arange(0, 10.0, 0.1)
y_true = np.sin(x_true).ravel() * 0.6

"""
[パラメーター]                [詳細]
kernel                      カーネル関数の選択。
                            rbf：ガウスカーネル(デフォルト)
                            linear：線形カーネル
                            poly：多項式カーネル
                            sigmoid：シグモイドカーネル
                            precomputed：カーネル
degree                      'poly'の場合のみ。次数を設定。
gamma                       {'rbf', 'poly', 'sigmoid'}の場合のみ。カーネル関数に使用される係数。
                            None：
coef0                       {'rbf', 'sigmoid'}の場合のみ重要。定数項の設定。
tol                         学習の終了条件？基本的に触らない。
C                           正則化パラメータ。正則化の強さはCに反比例する。
                            値は正。ペナルティはL2正則化。
epsilon                     不感損失関数(誤差の不感帯)の設定。ε-チューブの幅の半分。
shirnking                   ？処理短縮関連。
cache_size                  キャッシュサイズ(デフォルト200MB)。
verbose                     ログを表示する。
max_iter                    イテレータの制限をする(-1は制限なし)。

[Attributes]
support_                    
support_vectors_            
n_support_                  
dual_coef_                  
coef_                       
intercept_                  
"""
model = SVR(kernel="rbf",  # {"linear", "poly", "rbf", "sigmoid", "precomputed"}, default="rbf"
            degree=3,  # int, default=3
            gamma="scale",  # {"scale", "auto"} or float, default="scale"
            coef0=0.0,  # float, default=0.0
            tol=0.001,  # float, default=0.001
            C=1.0,  # float, default=1.0
            epsilon=0.1,  # float, default=0.1
            shrinking=True,  # bool, default=True
            cache_size=200,  # float, default=200
            verbose=False,  # bool, default=False
            max_iter=-1,  # int, default=-1
            )

model.fit(x, y)
y_pred = model.predict(x)
params = model.get_params()  # 設定パラメータの取得(辞書)
scores = model.score(x, y)  # 決定係数R^2
# params = model.set_params()  # 設定パラメータの設定(辞書)

supoort = model.support_
support_vectors = model.support_vectors_
dual_coef = model.dual_coef_
# coef = model.coef_  # 'liner'カーネルの場合のみ使用
fit_status = model.fit_status_
intercept = model.intercept_

# plot
plt.scatter(x, y, color="darkorange", label="data")
plt.plot(x_true, y_true, color="navy", label="sin")
plt.plot(x, y_pred, color="red", label="SVR(RBF)")
plt.legend()
plt.show()

print("end")

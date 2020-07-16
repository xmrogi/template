import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # 正解率
from sklearn.metrics import precision_score  # 精度
from sklearn.metrics import recall_score  # 検出率
from sklearn.metrics import f1_score  # F値


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                 sep=";",
                 encoding="utf-8"
                 )
print(df.head())

train_x = df.drop(['quality'], axis=1)
train_y = df['quality']
(train_x, test_x, train_y, test_y) = train_test_split(train_x, train_y, test_size=0.3)

"""
[パラメーター]                [詳細]
n_estimators                決定木の個数を指定。計算時間とトレードオフ。
criterion	                "gini"と"entropy"を指定できる．決定木はこの指標を基準にデータを分割する。
max_depth                   決定木の深さの最大値を指定。過学習を避けるためにはこれを調節するのが最も重要。
                            None：ノードを分けられるだけ分けるかmin_samples_split未満になるまでノードを構築する。
min_samples_split           ノードを分割するために必要な最小サンプルサイズ。
                            整数を指定した場合：その数。
                            小数を指定した場合：全サンプルサイズに対する割合個。
min_samples_leaf            葉を構成するのに必要な最小限のサンプルの数。
                            整数を指定した場合：その数。
                            小数を指定した場合：元々のサンプルサイズに占める割合と解釈される。
min_weight_fraction_leaf    葉における重みの総和の最小加重率を指定？
max_features                最適な分割をするために考慮する特徴量数を指定。
                            整数を指定した場合：その個数。
                            小数の場合：全特徴量に対する割合。
                            auto：特徴量数のルート個。
                            log2：log2(特徴量数)個。基本はNoneを使うべき。
                            None：n_features個。
max_leaf_nodes              最大の葉ノード数を指定。Noneの場合は制限なし。
min_impurity_split          決定木の成長の早期停止の閾値。不純度がこの値より大きいとき，ノードは分割される。
bootstrap                   決定木モデルの構築の際にブートストラップサンプリングを行うかどうかの指定。
oob_score                   学習における正確度の計算に OOB (out-of-bag)、すなわち、
                            各ブートストラップサンプリングでサンプリングされなかったサンプルを利用するかどうかの指定。
n_jobs                      フィットおよび予測の際に用いるスレッドの数を指定。-1 を指定した場合は計算機に載っている全スレッド分確保される。
random_state                乱数シードの指定。
verbose                     モデル構築の過程のメッセージを出すかどうか。
warm_start                  Trueを設定すると既に学習済みモデルに追加学習をする。
class_weight                各クラスに対する重み調整。
                            ディクショナリを指定する場合、{class_label：weight} の形式で，各クラスに重みを設定できる。
                            指定しない場合は全てのクラスに1が設定されている。
                            balanced を指定すると、n_samples / (n_classes * np.bincount(y))を計算し自動的調整する。
ccp_alpha                   Cost Complexity Pruning(決定木の複雑性に制約を付ける)で汎化性能を向上させる(過学習の防止)。
                            デフォルト(0.0)ではPruningを実行しない。
                            木を深く細分化するほど不純度は低下する一方、終端葉ノードの数が増えるので、そのバランスを取るαを見つければよい。
                            複雑性は以下の式で書ける。
                            R_α(T) = R(T) + α*|T|
                            R_α(T)：(Pruning後)終端葉ノードの誤分率の総和
                            R(T)：(Pruning前)終端葉ノードの誤分率の総和
                            |T|：終端葉ノードの数
                            α：Complexity Parameter
max_samples                 ブートストラップ法を使用する場合、入力データからサンプリングするデータサイズを指定。
                            None：入力データサイズ数。
                            int：max_samples。
                            float：max_samples * 入力データサイズ。(0, 1)の範囲で入力すること。

[Attributes]
classes_                    クラスラベルの配列
n_classes_                  クラスラベル数
n_features_                 特徴量数
n_outputs_                  出力データサイズ
feature_importances_        特徴量の(不純物ベースの)重要性を出力
oob_score_                  OBB誤り率
oob_decision_function_      OBB誤り率の決定関数を出力。
                            n_estimators が小さい場合、ブートストラップ中にデータを除外できなかった可能性がある。

"""
clf = RandomForestClassifier(n_estimators=30,  # int, default=100
                             criterion="gini",  # {"gini" or "entropy"}, fefault="gini"
                             max_depth=10,  # int, default=None
                             min_samples_split=2,  # int or float, default=2
                             min_samples_leaf=1,  # int float, default=1
                             min_weight_fraction_leaf=0.0,  # float, default=0.0
                             max_features="auto",  # {"auto", "sqrt", "log2"}, int or float, default="auto"
                             max_leaf_nodes=None,  # int, default=None
                             min_impurity_decrease=0.0,  # float, float=0.0
                             min_impurity_split=None,  # float, default=None
                             bootstrap=True,  # bool, default=True
                             oob_score=False,  # bool, default=False
                             n_jobs=-1,  # int, default=None
                             random_state=42,  # int or RandamState, default=None
                             warm_start=False,  # bool, default=False
                             class_weight=None,  # {"balanced", ""balanced_subsample}, dict or list of dicts, default=None
                             ccfp_alpha=0.0,  # non-negative float, default=0.0
                             max_samples=None,  # int or float, default=None
                             )
clf.fit(train_x, train_y)

y_pred = clf.predict(test_x)  #テスト用データの予測
print('Accuracy: {}'.format(accuracy_score(test_y, y_pred)))
print('Precision: {}'.format(precision_score(test_y, y_pred, average='micro')))
print('Recall: {}'.format(recall_score(test_y, y_pred, average='micro')))
print('F1: {}'.format(f1_score(test_y, y_pred, average='micro')))


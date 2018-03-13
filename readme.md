
# Train Model for Classifier for Categorical Training Dataset in Sklearn

## Objective

Regarding to machine learning classification, one of the common tasks is to load a csv file, specifies a classifier (SVM, KNN, tree, etc), select the training attributes (e.g. Outlook, Temperature, Humidity, Windy) and target attribute (e.g. Play golf). For training attributes with categorical values (e.g. Outlook has Rainy, Overcast, Sunny), we need to turn them from text into numeric values using sklearn.preprocessing.LabelEncoder.  
  
This Python TrainModel class helps to gather all the above procedures.


```python
import pandas as pd
df = pd.read_csv('weather_nominal.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Outlook</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Windy</th>
      <th>Play golf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rainy</td>
      <td>Hot</td>
      <td>High</td>
      <td>False</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rainy</td>
      <td>Hot</td>
      <td>High</td>
      <td>True</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>High</td>
      <td>False</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>High</td>
      <td>False</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sunny</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>False</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sunny</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>True</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Overcast</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>True</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Rainy</td>
      <td>Mild</td>
      <td>High</td>
      <td>False</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rainy</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>False</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>False</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Rainy</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>True</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Overcast</td>
      <td>Mild</td>
      <td>High</td>
      <td>True</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>Normal</td>
      <td>False</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>High</td>
      <td>True</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



## TrainModel


```python
# %load TrainModel.py
from sklearn import preprocessing
class TrainModel:
    def __init__(self, clf, df_raw, target):
        self.clf = clf
        self.encoders = {}
        self.df = self.transform(df_raw)
        self.target = target
    def transform(self, df_raw):
        df = df_raw.copy()
        for c in df:
            if (df[c].dtype=='object'):
                le = preprocessing.LabelEncoder()
                le.fit(df[c].tolist())
                result = le.transform(df[c].tolist())
                df[c] = result
                self.encoders[c] = le
        return df
    def get_train_x(self):
        return self.df[[x for x in self.df.columns if x!=self.target]]
    def get_train_y(self):
        return self.df[[self.target]].iloc[:,0].values
    def get_train_x_names(self):
        return [x for x in self.df.columns if x!=self.target]
    def get_train_y_names(self):
        return list(self.encoders[self.target].classes_)
    def run(self):
        self.clf.fit(self.get_train_x(), self.get_train_y())
    def predict(self):
        print('trained y', self.get_train_y())
        print('predict y', self.clf.predict(self.get_train_x()))

```

## Decision Tree

The usage of TrainMode is to supply a sklearn classifier, specifies the training attributes and target attribute.


```python
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]


## Linear Regression


```python
from sklearn import linear_model
clf = linear_model.LinearRegression()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0.44342746 0.1540409  0.69483934 0.38305745 0.6292113  0.33982473
     0.84264849 0.63446933 0.88062317 1.01129503 0.97332035 0.59649464
     1.32307692 0.09367089]


## Logistic Regression


```python
from sklearn import linear_model
clf = linear_model.LogisticRegression()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [1 0 1 0 1 1 1 1 1 1 1 1 1 0]


## Support Vector Machine


```python
from sklearn import svm
clf = svm.SVC()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [1 1 1 1 1 1 1 1 1 1 1 1 1 1]


## Naive Bayes

### Gaussian


```python
from sklearn import naive_bayes
clf = naive_bayes.GaussianNB()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0 0 1 0 1 1 1 0 1 1 1 1 1 0]


### Bernoulli


```python
from sklearn import naive_bayes
clf = naive_bayes.BernoulliNB()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0 0 1 0 1 1 1 0 1 1 1 1 1 0]


### MultinomialNB


```python
from sklearn import naive_bayes
clf = naive_bayes.MultinomialNB()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [1 0 1 1 1 1 1 1 1 1 1 1 1 0]


## K Nearest Neighbors


```python
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0 0 0 0 0 0 1 0 1 1 1 1 1 0]


## Random Forest


```python
from sklearn import ensemble
clf = ensemble.RandomForestClassifier()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]


## K Means Clustering


```python
from sklearn import cluster
clf = cluster.KMeans(n_clusters=3, random_state=0)
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0 0 0 1 2 2 0 1 2 1 1 0 0 1]


## Gradient Boosting


```python
from sklearn import ensemble
clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=0)
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]


## Extreme Gradient Boosting


```python
# git clone --recursive https://github.com/dmlc/xgboost
# cd xgboost; cp make/minimum.mk ./config.mk; make -j4
# cd python-package; sudo python setup.py install
from xgboost import XGBClassifier
clf = XGBClassifier()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0 0 0 1 1 1 1 1 1 1 1 0 1 0]


    /anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


## Light Gradient Boosting


```python
import lightgbm
train = TrainModel(clf, df, target=df.columns[-1])

# run(), does not call fit() but train()
d_train = lightgbm.Dataset(train.get_train_x(), label=train.get_train_y())
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 1
params['max_depth'] = 10
clf = lightgbm.train(params, d_train, 100)

train.predict()
```

    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0 0 0 1 1 1 1 1 1 1 1 0 1 0]


    /anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


## Catboost


```python
from catboost import CatBoostRegressor
clf = CatBoostRegressor()
train = TrainModel(clf, df, target=df.columns[-1])
train.run()
train.predict()
```

    0:	learn: 0.7952409	total: 55.4ms	remaining: 55.3s
    1:	learn: 0.7833689	total: 58.7ms	remaining: 29.3s
    2:	learn: 0.7761413	total: 60ms	remaining: 19.9s
    3:	learn: 0.7688179	total: 61.6ms	remaining: 15.3s
    4:	learn: 0.7581547	total: 62.7ms	remaining: 12.5s
    5:	learn: 0.7500263	total: 63.7ms	remaining: 10.6s
    6:	learn: 0.7392812	total: 64.9ms	remaining: 9.21s
    7:	learn: 0.7288340	total: 66.5ms	remaining: 8.24s
    8:	learn: 0.7197826	total: 67.4ms	remaining: 7.42s
    9:	learn: 0.7136919	total: 69.4ms	remaining: 6.87s
    10:	learn: 0.7055841	total: 71.3ms	remaining: 6.41s
    11:	learn: 0.7002862	total: 74.9ms	remaining: 6.17s
    12:	learn: 0.6928357	total: 76.4ms	remaining: 5.8s
    13:	learn: 0.6867423	total: 77.4ms	remaining: 5.45s
    14:	learn: 0.6788206	total: 78.3ms	remaining: 5.14s
    15:	learn: 0.6713538	total: 79.2ms	remaining: 4.87s
    16:	learn: 0.6630735	total: 80.1ms	remaining: 4.63s
    17:	learn: 0.6561555	total: 81.1ms	remaining: 4.43s
    18:	learn: 0.6504448	total: 82.3ms	remaining: 4.25s
    19:	learn: 0.6451427	total: 83.7ms	remaining: 4.1s
    20:	learn: 0.6391165	total: 84.9ms	remaining: 3.96s
    21:	learn: 0.6316066	total: 85.9ms	remaining: 3.82s
    22:	learn: 0.6246751	total: 86.9ms	remaining: 3.69s
    23:	learn: 0.6175151	total: 87.8ms	remaining: 3.57s
    24:	learn: 0.6102728	total: 88.8ms	remaining: 3.46s
    25:	learn: 0.6036734	total: 89.9ms	remaining: 3.37s
    26:	learn: 0.5974046	total: 90.7ms	remaining: 3.27s
    27:	learn: 0.5919939	total: 91.7ms	remaining: 3.18s
    28:	learn: 0.5849701	total: 92.5ms	remaining: 3.1s
    29:	learn: 0.5791654	total: 93.2ms	remaining: 3.01s
    30:	learn: 0.5740217	total: 94.3ms	remaining: 2.95s
    31:	learn: 0.5682479	total: 95.2ms	remaining: 2.88s
    32:	learn: 0.5618102	total: 96.2ms	remaining: 2.82s
    33:	learn: 0.5555701	total: 97ms	remaining: 2.76s
    34:	learn: 0.5500662	total: 97.8ms	remaining: 2.7s
    35:	learn: 0.5442954	total: 99.2ms	remaining: 2.65s
    36:	learn: 0.5387842	total: 100ms	remaining: 2.61s
    37:	learn: 0.5340243	total: 102ms	remaining: 2.57s
    38:	learn: 0.5294958	total: 103ms	remaining: 2.53s
    39:	learn: 0.5249597	total: 104ms	remaining: 2.49s
    40:	learn: 0.5207700	total: 105ms	remaining: 2.45s
    41:	learn: 0.5166386	total: 106ms	remaining: 2.41s
    42:	learn: 0.5131428	total: 107ms	remaining: 2.38s
    43:	learn: 0.5094080	total: 108ms	remaining: 2.35s
    44:	learn: 0.5050933	total: 109ms	remaining: 2.31s
    45:	learn: 0.5014404	total: 110ms	remaining: 2.28s
    46:	learn: 0.4971827	total: 111ms	remaining: 2.25s
    47:	learn: 0.4933710	total: 112ms	remaining: 2.23s
    48:	learn: 0.4895858	total: 113ms	remaining: 2.2s
    49:	learn: 0.4859293	total: 114ms	remaining: 2.17s
    50:	learn: 0.4829335	total: 115ms	remaining: 2.15s
    51:	learn: 0.4782890	total: 116ms	remaining: 2.12s
    52:	learn: 0.4746170	total: 117ms	remaining: 2.1s
    53:	learn: 0.4705704	total: 118ms	remaining: 2.07s
    54:	learn: 0.4663469	total: 119ms	remaining: 2.04s
    55:	learn: 0.4629243	total: 120ms	remaining: 2.02s
    56:	learn: 0.4594166	total: 121ms	remaining: 2s
    57:	learn: 0.4551876	total: 122ms	remaining: 1.98s
    58:	learn: 0.4518181	total: 123ms	remaining: 1.96s
    59:	learn: 0.4482530	total: 124ms	remaining: 1.94s
    60:	learn: 0.4452534	total: 125ms	remaining: 1.92s
    61:	learn: 0.4425089	total: 126ms	remaining: 1.9s
    62:	learn: 0.4388489	total: 126ms	remaining: 1.88s
    63:	learn: 0.4366314	total: 127ms	remaining: 1.86s
    64:	learn: 0.4343339	total: 129ms	remaining: 1.85s
    65:	learn: 0.4318490	total: 130ms	remaining: 1.83s
    66:	learn: 0.4286812	total: 131ms	remaining: 1.82s
    67:	learn: 0.4261624	total: 131ms	remaining: 1.8s
    68:	learn: 0.4225742	total: 133ms	remaining: 1.79s
    69:	learn: 0.4190274	total: 133ms	remaining: 1.77s
    70:	learn: 0.4161214	total: 135ms	remaining: 1.76s
    71:	learn: 0.4136151	total: 136ms	remaining: 1.75s
    72:	learn: 0.4102358	total: 137ms	remaining: 1.73s
    73:	learn: 0.4078176	total: 138ms	remaining: 1.72s
    74:	learn: 0.4050178	total: 139ms	remaining: 1.72s
    75:	learn: 0.4021830	total: 140ms	remaining: 1.71s
    76:	learn: 0.3989767	total: 141ms	remaining: 1.69s
    77:	learn: 0.3961940	total: 142ms	remaining: 1.68s
    78:	learn: 0.3928555	total: 144ms	remaining: 1.68s
    79:	learn: 0.3904134	total: 144ms	remaining: 1.66s
    80:	learn: 0.3884814	total: 146ms	remaining: 1.65s
    81:	learn: 0.3857195	total: 147ms	remaining: 1.64s
    82:	learn: 0.3834769	total: 148ms	remaining: 1.63s
    83:	learn: 0.3817709	total: 149ms	remaining: 1.62s
    84:	learn: 0.3783727	total: 150ms	remaining: 1.61s
    85:	learn: 0.3757989	total: 151ms	remaining: 1.61s
    86:	learn: 0.3733179	total: 152ms	remaining: 1.6s
    87:	learn: 0.3705754	total: 154ms	remaining: 1.59s
    88:	learn: 0.3685946	total: 155ms	remaining: 1.58s
    89:	learn: 0.3665179	total: 156ms	remaining: 1.57s
    90:	learn: 0.3651023	total: 157ms	remaining: 1.57s
    91:	learn: 0.3627661	total: 158ms	remaining: 1.56s
    92:	learn: 0.3613692	total: 159ms	remaining: 1.55s
    93:	learn: 0.3585904	total: 161ms	remaining: 1.55s
    94:	learn: 0.3565219	total: 161ms	remaining: 1.54s
    95:	learn: 0.3541075	total: 163ms	remaining: 1.53s
    96:	learn: 0.3517172	total: 164ms	remaining: 1.53s
    97:	learn: 0.3503233	total: 165ms	remaining: 1.52s
    98:	learn: 0.3489360	total: 166ms	remaining: 1.51s
    99:	learn: 0.3475827	total: 167ms	remaining: 1.5s
    100:	learn: 0.3464613	total: 168ms	remaining: 1.5s
    101:	learn: 0.3449297	total: 169ms	remaining: 1.49s
    102:	learn: 0.3425914	total: 170ms	remaining: 1.48s
    103:	learn: 0.3408064	total: 172ms	remaining: 1.48s
    104:	learn: 0.3383396	total: 173ms	remaining: 1.47s
    105:	learn: 0.3371877	total: 174ms	remaining: 1.47s
    106:	learn: 0.3358269	total: 175ms	remaining: 1.46s
    107:	learn: 0.3341192	total: 176ms	remaining: 1.45s
    108:	learn: 0.3326052	total: 177ms	remaining: 1.45s
    109:	learn: 0.3314814	total: 178ms	remaining: 1.44s
    110:	learn: 0.3293329	total: 179ms	remaining: 1.43s
    111:	learn: 0.3275154	total: 180ms	remaining: 1.43s
    112:	learn: 0.3259138	total: 181ms	remaining: 1.42s
    113:	learn: 0.3231342	total: 182ms	remaining: 1.42s
    114:	learn: 0.3212549	total: 183ms	remaining: 1.41s
    115:	learn: 0.3203262	total: 184ms	remaining: 1.4s
    116:	learn: 0.3193103	total: 185ms	remaining: 1.4s
    117:	learn: 0.3166526	total: 186ms	remaining: 1.39s
    118:	learn: 0.3139380	total: 188ms	remaining: 1.39s
    119:	learn: 0.3130925	total: 189ms	remaining: 1.39s
    120:	learn: 0.3105081	total: 190ms	remaining: 1.38s
    121:	learn: 0.3077623	total: 191ms	remaining: 1.37s
    122:	learn: 0.3067819	total: 192ms	remaining: 1.37s
    123:	learn: 0.3051320	total: 193ms	remaining: 1.36s
    124:	learn: 0.3026456	total: 194ms	remaining: 1.36s
    125:	learn: 0.3007603	total: 195ms	remaining: 1.35s
    126:	learn: 0.2981950	total: 197ms	remaining: 1.35s
    127:	learn: 0.2970398	total: 198ms	remaining: 1.35s
    128:	learn: 0.2946048	total: 201ms	remaining: 1.36s
    129:	learn: 0.2922321	total: 203ms	remaining: 1.36s
    130:	learn: 0.2908540	total: 205ms	remaining: 1.36s
    131:	learn: 0.2893864	total: 207ms	remaining: 1.36s
    132:	learn: 0.2881613	total: 208ms	remaining: 1.36s
    133:	learn: 0.2858951	total: 210ms	remaining: 1.35s
    134:	learn: 0.2851484	total: 211ms	remaining: 1.35s
    135:	learn: 0.2843728	total: 215ms	remaining: 1.37s
    136:	learn: 0.2829746	total: 216ms	remaining: 1.36s
    137:	learn: 0.2823854	total: 217ms	remaining: 1.36s
    138:	learn: 0.2804088	total: 218ms	remaining: 1.35s
    139:	learn: 0.2788830	total: 219ms	remaining: 1.35s
    140:	learn: 0.2773773	total: 220ms	remaining: 1.34s
    141:	learn: 0.2759077	total: 222ms	remaining: 1.34s
    142:	learn: 0.2739089	total: 224ms	remaining: 1.34s
    143:	learn: 0.2731264	total: 226ms	remaining: 1.34s
    144:	learn: 0.2717083	total: 227ms	remaining: 1.34s
    145:	learn: 0.2711512	total: 228ms	remaining: 1.33s
    146:	learn: 0.2688967	total: 230ms	remaining: 1.33s
    147:	learn: 0.2679964	total: 231ms	remaining: 1.33s
    148:	learn: 0.2659763	total: 232ms	remaining: 1.32s
    149:	learn: 0.2639815	total: 237ms	remaining: 1.34s
    150:	learn: 0.2631294	total: 239ms	remaining: 1.34s
    151:	learn: 0.2608852	total: 247ms	remaining: 1.38s
    152:	learn: 0.2586132	total: 248ms	remaining: 1.38s
    153:	learn: 0.2564779	total: 256ms	remaining: 1.4s
    154:	learn: 0.2554793	total: 257ms	remaining: 1.4s
    155:	learn: 0.2550659	total: 259ms	remaining: 1.4s
    156:	learn: 0.2538150	total: 261ms	remaining: 1.4s
    157:	learn: 0.2519438	total: 263ms	remaining: 1.4s
    158:	learn: 0.2507462	total: 264ms	remaining: 1.4s
    159:	learn: 0.2495779	total: 266ms	remaining: 1.39s
    160:	learn: 0.2491204	total: 267ms	remaining: 1.39s
    161:	learn: 0.2486812	total: 268ms	remaining: 1.39s
    162:	learn: 0.2475653	total: 269ms	remaining: 1.38s
    163:	learn: 0.2454382	total: 271ms	remaining: 1.38s
    164:	learn: 0.2445914	total: 273ms	remaining: 1.38s
    165:	learn: 0.2439438	total: 274ms	remaining: 1.38s
    166:	learn: 0.2419629	total: 276ms	remaining: 1.38s
    167:	learn: 0.2415771	total: 278ms	remaining: 1.38s
    168:	learn: 0.2413275	total: 279ms	remaining: 1.37s
    169:	learn: 0.2400646	total: 281ms	remaining: 1.37s
    170:	learn: 0.2393828	total: 282ms	remaining: 1.37s
    171:	learn: 0.2386218	total: 284ms	remaining: 1.36s
    172:	learn: 0.2376302	total: 285ms	remaining: 1.36s
    173:	learn: 0.2367211	total: 286ms	remaining: 1.36s
    174:	learn: 0.2352049	total: 289ms	remaining: 1.36s
    175:	learn: 0.2332022	total: 290ms	remaining: 1.36s
    176:	learn: 0.2328815	total: 291ms	remaining: 1.35s
    177:	learn: 0.2325738	total: 292ms	remaining: 1.35s
    178:	learn: 0.2316462	total: 294ms	remaining: 1.35s
    179:	learn: 0.2309708	total: 295ms	remaining: 1.34s
    180:	learn: 0.2300804	total: 297ms	remaining: 1.34s
    181:	learn: 0.2293639	total: 299ms	remaining: 1.34s
    182:	learn: 0.2288411	total: 300ms	remaining: 1.34s
    183:	learn: 0.2282096	total: 301ms	remaining: 1.33s
    184:	learn: 0.2275980	total: 302ms	remaining: 1.33s
    185:	learn: 0.2260043	total: 304ms	remaining: 1.33s
    186:	learn: 0.2246373	total: 305ms	remaining: 1.33s
    187:	learn: 0.2243919	total: 306ms	remaining: 1.32s
    188:	learn: 0.2235683	total: 307ms	remaining: 1.32s
    189:	learn: 0.2229994	total: 308ms	remaining: 1.31s
    190:	learn: 0.2226779	total: 309ms	remaining: 1.31s
    191:	learn: 0.2224117	total: 310ms	remaining: 1.3s
    192:	learn: 0.2209237	total: 311ms	remaining: 1.3s
    193:	learn: 0.2195177	total: 312ms	remaining: 1.3s
    194:	learn: 0.2187378	total: 314ms	remaining: 1.29s
    195:	learn: 0.2168799	total: 315ms	remaining: 1.29s
    196:	learn: 0.2166825	total: 316ms	remaining: 1.29s
    197:	learn: 0.2155131	total: 317ms	remaining: 1.28s
    198:	learn: 0.2136840	total: 318ms	remaining: 1.28s
    199:	learn: 0.2122845	total: 319ms	remaining: 1.28s
    200:	learn: 0.2111442	total: 321ms	remaining: 1.27s
    201:	learn: 0.2104347	total: 322ms	remaining: 1.27s
    202:	learn: 0.2086523	total: 323ms	remaining: 1.27s
    203:	learn: 0.2069177	total: 324ms	remaining: 1.26s
    204:	learn: 0.2055947	total: 325ms	remaining: 1.26s
    205:	learn: 0.2054281	total: 326ms	remaining: 1.26s
    206:	learn: 0.2051975	total: 327ms	remaining: 1.25s
    207:	learn: 0.2042054	total: 328ms	remaining: 1.25s
    208:	learn: 0.2041023	total: 329ms	remaining: 1.25s
    209:	learn: 0.2025180	total: 331ms	remaining: 1.25s
    210:	learn: 0.2011209	total: 332ms	remaining: 1.24s
    211:	learn: 0.2008926	total: 333ms	remaining: 1.24s
    212:	learn: 0.1997035	total: 334ms	remaining: 1.23s
    213:	learn: 0.1996136	total: 335ms	remaining: 1.23s
    214:	learn: 0.1984416	total: 336ms	remaining: 1.23s
    215:	learn: 0.1983569	total: 337ms	remaining: 1.22s
    216:	learn: 0.1979098	total: 338ms	remaining: 1.22s
    217:	learn: 0.1962334	total: 339ms	remaining: 1.22s
    218:	learn: 0.1961544	total: 340ms	remaining: 1.21s
    219:	learn: 0.1959253	total: 341ms	remaining: 1.21s
    220:	learn: 0.1943379	total: 342ms	remaining: 1.21s
    221:	learn: 0.1941160	total: 343ms	remaining: 1.2s
    222:	learn: 0.1936026	total: 344ms	remaining: 1.2s
    223:	learn: 0.1923845	total: 345ms	remaining: 1.2s
    224:	learn: 0.1911177	total: 346ms	remaining: 1.19s
    225:	learn: 0.1910512	total: 347ms	remaining: 1.19s
    226:	learn: 0.1898103	total: 348ms	remaining: 1.19s
    227:	learn: 0.1897470	total: 349ms	remaining: 1.18s
    228:	learn: 0.1896466	total: 350ms	remaining: 1.18s
    229:	learn: 0.1889768	total: 351ms	remaining: 1.18s
    230:	learn: 0.1878767	total: 352ms	remaining: 1.17s
    231:	learn: 0.1866131	total: 353ms	remaining: 1.17s
    232:	learn: 0.1854432	total: 354ms	remaining: 1.17s
    233:	learn: 0.1853885	total: 355ms	remaining: 1.16s
    234:	learn: 0.1853362	total: 356ms	remaining: 1.16s
    235:	learn: 0.1847429	total: 357ms	remaining: 1.16s
    236:	learn: 0.1832746	total: 358ms	remaining: 1.15s
    237:	learn: 0.1817207	total: 359ms	remaining: 1.15s
    238:	learn: 0.1816383	total: 360ms	remaining: 1.15s
    239:	learn: 0.1804495	total: 361ms	remaining: 1.14s
    240:	learn: 0.1792849	total: 362ms	remaining: 1.14s
    241:	learn: 0.1777691	total: 363ms	remaining: 1.14s
    242:	learn: 0.1772247	total: 364ms	remaining: 1.14s
    243:	learn: 0.1760393	total: 365ms	remaining: 1.13s
    244:	learn: 0.1759946	total: 366ms	remaining: 1.13s
    245:	learn: 0.1755892	total: 367ms	remaining: 1.13s
    246:	learn: 0.1754401	total: 368ms	remaining: 1.12s
    247:	learn: 0.1753996	total: 369ms	remaining: 1.12s
    248:	learn: 0.1740927	total: 371ms	remaining: 1.12s
    249:	learn: 0.1730034	total: 372ms	remaining: 1.12s
    250:	learn: 0.1715421	total: 373ms	remaining: 1.11s
    251:	learn: 0.1705631	total: 374ms	remaining: 1.11s
    252:	learn: 0.1692343	total: 375ms	remaining: 1.11s
    253:	learn: 0.1679308	total: 377ms	remaining: 1.11s
    254:	learn: 0.1666957	total: 379ms	remaining: 1.1s
    255:	learn: 0.1662811	total: 379ms	remaining: 1.1s
    256:	learn: 0.1662244	total: 380ms	remaining: 1.1s
    257:	learn: 0.1661701	total: 381ms	remaining: 1.1s
    258:	learn: 0.1661372	total: 382ms	remaining: 1.09s
    259:	learn: 0.1649245	total: 383ms	remaining: 1.09s
    260:	learn: 0.1638526	total: 385ms	remaining: 1.09s
    261:	learn: 0.1638218	total: 386ms	remaining: 1.09s
    262:	learn: 0.1633328	total: 387ms	remaining: 1.08s
    263:	learn: 0.1620965	total: 388ms	remaining: 1.08s
    264:	learn: 0.1610453	total: 389ms	remaining: 1.08s
    265:	learn: 0.1607228	total: 390ms	remaining: 1.08s
    266:	learn: 0.1604469	total: 391ms	remaining: 1.07s
    267:	learn: 0.1595420	total: 392ms	remaining: 1.07s
    268:	learn: 0.1583494	total: 394ms	remaining: 1.07s
    269:	learn: 0.1572093	total: 395ms	remaining: 1.07s
    270:	learn: 0.1567686	total: 397ms	remaining: 1.07s
    271:	learn: 0.1563397	total: 398ms	remaining: 1.06s
    272:	learn: 0.1563021	total: 399ms	remaining: 1.06s
    273:	learn: 0.1562660	total: 400ms	remaining: 1.06s
    274:	learn: 0.1551490	total: 401ms	remaining: 1.06s
    275:	learn: 0.1541874	total: 402ms	remaining: 1.05s
    276:	learn: 0.1537623	total: 403ms	remaining: 1.05s
    277:	learn: 0.1526696	total: 405ms	remaining: 1.05s
    278:	learn: 0.1515622	total: 406ms	remaining: 1.05s
    279:	learn: 0.1502934	total: 408ms	remaining: 1.05s
    280:	learn: 0.1501763	total: 409ms	remaining: 1.04s
    281:	learn: 0.1498370	total: 410ms	remaining: 1.04s
    282:	learn: 0.1498167	total: 411ms	remaining: 1.04s
    283:	learn: 0.1489876	total: 412ms	remaining: 1.04s
    284:	learn: 0.1479437	total: 414ms	remaining: 1.04s
    285:	learn: 0.1479250	total: 415ms	remaining: 1.04s
    286:	learn: 0.1477018	total: 416ms	remaining: 1.03s
    287:	learn: 0.1475737	total: 419ms	remaining: 1.03s
    288:	learn: 0.1474509	total: 420ms	remaining: 1.03s
    289:	learn: 0.1474267	total: 421ms	remaining: 1.03s
    290:	learn: 0.1471164	total: 422ms	remaining: 1.03s
    291:	learn: 0.1467309	total: 423ms	remaining: 1.02s
    292:	learn: 0.1463650	total: 424ms	remaining: 1.02s
    293:	learn: 0.1463504	total: 425ms	remaining: 1.02s
    294:	learn: 0.1459783	total: 426ms	remaining: 1.02s
    295:	learn: 0.1459646	total: 427ms	remaining: 1.01s
    296:	learn: 0.1450251	total: 428ms	remaining: 1.01s
    297:	learn: 0.1445986	total: 429ms	remaining: 1.01s
    298:	learn: 0.1445857	total: 430ms	remaining: 1.01s
    299:	learn: 0.1445653	total: 432ms	remaining: 1.01s
    300:	learn: 0.1436701	total: 433ms	remaining: 1s
    301:	learn: 0.1433295	total: 434ms	remaining: 1s
    302:	learn: 0.1433107	total: 435ms	remaining: 1s
    303:	learn: 0.1424323	total: 437ms	remaining: 1s
    304:	learn: 0.1415140	total: 438ms	remaining: 998ms
    305:	learn: 0.1412406	total: 439ms	remaining: 995ms
    306:	learn: 0.1402720	total: 440ms	remaining: 994ms
    307:	learn: 0.1400982	total: 441ms	remaining: 992ms
    308:	learn: 0.1393214	total: 442ms	remaining: 989ms
    309:	learn: 0.1390047	total: 443ms	remaining: 987ms
    310:	learn: 0.1386721	total: 445ms	remaining: 985ms
    311:	learn: 0.1386567	total: 446ms	remaining: 983ms
    312:	learn: 0.1374976	total: 447ms	remaining: 982ms
    313:	learn: 0.1367374	total: 449ms	remaining: 981ms
    314:	learn: 0.1367234	total: 450ms	remaining: 979ms
    315:	learn: 0.1355820	total: 452ms	remaining: 979ms
    316:	learn: 0.1352646	total: 453ms	remaining: 977ms
    317:	learn: 0.1344526	total: 455ms	remaining: 976ms
    318:	learn: 0.1336504	total: 457ms	remaining: 976ms
    319:	learn: 0.1328021	total: 459ms	remaining: 976ms
    320:	learn: 0.1316972	total: 461ms	remaining: 975ms
    321:	learn: 0.1306010	total: 463ms	remaining: 975ms
    322:	learn: 0.1305917	total: 465ms	remaining: 975ms
    323:	learn: 0.1305805	total: 466ms	remaining: 973ms
    324:	learn: 0.1305719	total: 468ms	remaining: 971ms
    325:	learn: 0.1297734	total: 469ms	remaining: 971ms
    326:	learn: 0.1297631	total: 472ms	remaining: 972ms
    327:	learn: 0.1288953	total: 476ms	remaining: 976ms
    328:	learn: 0.1280225	total: 480ms	remaining: 980ms
    329:	learn: 0.1277359	total: 482ms	remaining: 978ms
    330:	learn: 0.1274570	total: 483ms	remaining: 976ms
    331:	learn: 0.1266595	total: 484ms	remaining: 975ms
    332:	learn: 0.1259033	total: 486ms	remaining: 973ms
    333:	learn: 0.1256772	total: 487ms	remaining: 971ms
    334:	learn: 0.1253518	total: 489ms	remaining: 970ms
    335:	learn: 0.1245043	total: 490ms	remaining: 968ms
    336:	learn: 0.1237408	total: 491ms	remaining: 966ms
    337:	learn: 0.1229932	total: 492ms	remaining: 964ms
    338:	learn: 0.1229104	total: 493ms	remaining: 962ms
    339:	learn: 0.1226667	total: 494ms	remaining: 960ms
    340:	learn: 0.1224029	total: 496ms	remaining: 958ms
    341:	learn: 0.1217159	total: 497ms	remaining: 956ms
    342:	learn: 0.1215903	total: 498ms	remaining: 954ms
    343:	learn: 0.1215833	total: 499ms	remaining: 952ms
    344:	learn: 0.1212810	total: 500ms	remaining: 950ms
    345:	learn: 0.1210340	total: 501ms	remaining: 948ms
    346:	learn: 0.1207944	total: 503ms	remaining: 946ms
    347:	learn: 0.1207880	total: 504ms	remaining: 944ms
    348:	learn: 0.1202606	total: 505ms	remaining: 942ms
    349:	learn: 0.1194774	total: 507ms	remaining: 942ms
    350:	learn: 0.1192187	total: 509ms	remaining: 941ms
    351:	learn: 0.1189379	total: 510ms	remaining: 939ms
    352:	learn: 0.1182344	total: 511ms	remaining: 936ms
    353:	learn: 0.1182289	total: 512ms	remaining: 934ms
    354:	learn: 0.1176595	total: 513ms	remaining: 932ms
    355:	learn: 0.1174503	total: 514ms	remaining: 930ms
    356:	learn: 0.1173464	total: 515ms	remaining: 927ms
    357:	learn: 0.1166802	total: 516ms	remaining: 925ms
    358:	learn: 0.1159621	total: 517ms	remaining: 923ms
    359:	learn: 0.1152597	total: 518ms	remaining: 921ms
    360:	learn: 0.1150477	total: 519ms	remaining: 918ms
    361:	learn: 0.1148458	total: 520ms	remaining: 916ms
    362:	learn: 0.1141811	total: 521ms	remaining: 914ms
    363:	learn: 0.1139789	total: 522ms	remaining: 911ms
    364:	learn: 0.1132958	total: 523ms	remaining: 910ms
    365:	learn: 0.1126616	total: 524ms	remaining: 908ms
    366:	learn: 0.1126572	total: 525ms	remaining: 905ms
    367:	learn: 0.1126538	total: 526ms	remaining: 903ms
    368:	learn: 0.1125808	total: 527ms	remaining: 901ms
    369:	learn: 0.1123892	total: 528ms	remaining: 898ms
    370:	learn: 0.1117270	total: 529ms	remaining: 897ms
    371:	learn: 0.1110788	total: 530ms	remaining: 895ms
    372:	learn: 0.1109001	total: 531ms	remaining: 893ms
    373:	learn: 0.1102734	total: 532ms	remaining: 891ms
    374:	learn: 0.1095924	total: 534ms	remaining: 889ms
    375:	learn: 0.1088642	total: 535ms	remaining: 888ms
    376:	learn: 0.1086277	total: 536ms	remaining: 886ms
    377:	learn: 0.1085471	total: 537ms	remaining: 883ms
    378:	learn: 0.1079218	total: 538ms	remaining: 881ms
    379:	learn: 0.1079191	total: 539ms	remaining: 879ms
    380:	learn: 0.1077336	total: 540ms	remaining: 877ms
    381:	learn: 0.1072816	total: 541ms	remaining: 876ms
    382:	learn: 0.1072779	total: 542ms	remaining: 874ms
    383:	learn: 0.1072175	total: 544ms	remaining: 872ms
    384:	learn: 0.1066146	total: 545ms	remaining: 870ms
    385:	learn: 0.1059589	total: 546ms	remaining: 868ms
    386:	learn: 0.1058813	total: 547ms	remaining: 866ms
    387:	learn: 0.1051898	total: 548ms	remaining: 864ms
    388:	learn: 0.1051863	total: 549ms	remaining: 862ms
    389:	learn: 0.1050063	total: 550ms	remaining: 860ms
    390:	learn: 0.1049334	total: 551ms	remaining: 859ms
    391:	learn: 0.1047745	total: 552ms	remaining: 857ms
    392:	learn: 0.1041287	total: 554ms	remaining: 856ms
    393:	learn: 0.1039187	total: 555ms	remaining: 854ms
    394:	learn: 0.1033297	total: 556ms	remaining: 852ms
    395:	learn: 0.1031244	total: 558ms	remaining: 851ms
    396:	learn: 0.1031213	total: 559ms	remaining: 850ms
    397:	learn: 0.1029762	total: 561ms	remaining: 848ms
    398:	learn: 0.1023135	total: 563ms	remaining: 848ms
    399:	learn: 0.1021733	total: 565ms	remaining: 847ms
    400:	learn: 0.1020135	total: 566ms	remaining: 846ms
    401:	learn: 0.1018207	total: 567ms	remaining: 844ms
    402:	learn: 0.1012388	total: 569ms	remaining: 842ms
    403:	learn: 0.1011743	total: 570ms	remaining: 840ms
    404:	learn: 0.1003406	total: 572ms	remaining: 841ms
    405:	learn: 0.1003388	total: 575ms	remaining: 841ms
    406:	learn: 0.1002067	total: 576ms	remaining: 839ms
    407:	learn: 0.0995760	total: 577ms	remaining: 837ms
    408:	learn: 0.0995731	total: 578ms	remaining: 835ms
    409:	learn: 0.0989752	total: 580ms	remaining: 834ms
    410:	learn: 0.0989736	total: 581ms	remaining: 832ms
    411:	learn: 0.0984130	total: 582ms	remaining: 830ms
    412:	learn: 0.0978534	total: 583ms	remaining: 829ms
    413:	learn: 0.0977923	total: 586ms	remaining: 830ms
    414:	learn: 0.0972613	total: 588ms	remaining: 828ms
    415:	learn: 0.0970838	total: 589ms	remaining: 826ms
    416:	learn: 0.0965092	total: 590ms	remaining: 825ms
    417:	learn: 0.0959562	total: 592ms	remaining: 824ms
    418:	learn: 0.0958237	total: 593ms	remaining: 822ms
    419:	learn: 0.0954174	total: 594ms	remaining: 821ms
    420:	learn: 0.0954158	total: 595ms	remaining: 818ms
    421:	learn: 0.0953231	total: 596ms	remaining: 817ms
    422:	learn: 0.0951853	total: 597ms	remaining: 815ms
    423:	learn: 0.0950694	total: 599ms	remaining: 814ms
    424:	learn: 0.0944797	total: 600ms	remaining: 812ms
    425:	learn: 0.0943136	total: 601ms	remaining: 810ms
    426:	learn: 0.0941899	total: 602ms	remaining: 808ms
    427:	learn: 0.0936244	total: 604ms	remaining: 807ms
    428:	learn: 0.0935103	total: 605ms	remaining: 805ms
    429:	learn: 0.0935082	total: 606ms	remaining: 803ms
    430:	learn: 0.0935062	total: 607ms	remaining: 801ms
    431:	learn: 0.0933495	total: 608ms	remaining: 799ms
    432:	learn: 0.0932395	total: 609ms	remaining: 797ms
    433:	learn: 0.0927157	total: 610ms	remaining: 796ms
    434:	learn: 0.0926052	total: 612ms	remaining: 794ms
    435:	learn: 0.0920612	total: 613ms	remaining: 793ms
    436:	learn: 0.0913233	total: 615ms	remaining: 792ms
    437:	learn: 0.0908557	total: 616ms	remaining: 790ms
    438:	learn: 0.0908085	total: 617ms	remaining: 788ms
    439:	learn: 0.0902848	total: 619ms	remaining: 787ms
    440:	learn: 0.0902483	total: 619ms	remaining: 785ms
    441:	learn: 0.0897369	total: 620ms	remaining: 783ms
    442:	learn: 0.0896277	total: 621ms	remaining: 781ms
    443:	learn: 0.0891264	total: 623ms	remaining: 780ms
    444:	learn: 0.0886012	total: 624ms	remaining: 778ms
    445:	learn: 0.0881032	total: 626ms	remaining: 778ms
    446:	learn: 0.0880581	total: 627ms	remaining: 775ms
    447:	learn: 0.0879853	total: 628ms	remaining: 774ms
    448:	learn: 0.0874967	total: 629ms	remaining: 772ms
    449:	learn: 0.0869956	total: 630ms	remaining: 770ms
    450:	learn: 0.0869935	total: 631ms	remaining: 768ms
    451:	learn: 0.0868952	total: 632ms	remaining: 767ms
    452:	learn: 0.0868932	total: 633ms	remaining: 765ms
    453:	learn: 0.0868522	total: 634ms	remaining: 763ms
    454:	learn: 0.0863594	total: 635ms	remaining: 761ms
    455:	learn: 0.0858770	total: 636ms	remaining: 759ms
    456:	learn: 0.0854105	total: 637ms	remaining: 757ms
    457:	learn: 0.0854085	total: 639ms	remaining: 756ms
    458:	learn: 0.0849334	total: 640ms	remaining: 754ms
    459:	learn: 0.0849040	total: 641ms	remaining: 752ms
    460:	learn: 0.0848619	total: 642ms	remaining: 750ms
    461:	learn: 0.0848217	total: 642ms	remaining: 748ms
    462:	learn: 0.0843559	total: 643ms	remaining: 746ms
    463:	learn: 0.0838704	total: 645ms	remaining: 745ms
    464:	learn: 0.0832072	total: 646ms	remaining: 743ms
    465:	learn: 0.0831188	total: 647ms	remaining: 742ms
    466:	learn: 0.0831166	total: 648ms	remaining: 740ms
    467:	learn: 0.0827426	total: 650ms	remaining: 738ms
    468:	learn: 0.0822926	total: 651ms	remaining: 737ms
    469:	learn: 0.0822536	total: 652ms	remaining: 736ms
    470:	learn: 0.0817839	total: 654ms	remaining: 734ms
    471:	learn: 0.0817473	total: 655ms	remaining: 732ms
    472:	learn: 0.0816640	total: 656ms	remaining: 730ms
    473:	learn: 0.0816376	total: 656ms	remaining: 728ms
    474:	learn: 0.0816022	total: 657ms	remaining: 726ms
    475:	learn: 0.0815683	total: 658ms	remaining: 724ms
    476:	learn: 0.0815357	total: 660ms	remaining: 723ms
    477:	learn: 0.0811274	total: 661ms	remaining: 721ms
    478:	learn: 0.0811252	total: 662ms	remaining: 720ms
    479:	learn: 0.0810937	total: 663ms	remaining: 718ms
    480:	learn: 0.0806491	total: 664ms	remaining: 716ms
    481:	learn: 0.0806218	total: 665ms	remaining: 714ms
    482:	learn: 0.0801904	total: 667ms	remaining: 714ms
    483:	learn: 0.0800736	total: 668ms	remaining: 712ms
    484:	learn: 0.0796483	total: 670ms	remaining: 711ms
    485:	learn: 0.0795572	total: 671ms	remaining: 710ms
    486:	learn: 0.0794805	total: 673ms	remaining: 709ms
    487:	learn: 0.0794779	total: 674ms	remaining: 707ms
    488:	learn: 0.0790385	total: 675ms	remaining: 705ms
    489:	learn: 0.0786250	total: 676ms	remaining: 704ms
    490:	learn: 0.0786021	total: 677ms	remaining: 702ms
    491:	learn: 0.0785992	total: 678ms	remaining: 700ms
    492:	learn: 0.0785965	total: 679ms	remaining: 698ms
    493:	learn: 0.0785226	total: 680ms	remaining: 697ms
    494:	learn: 0.0781762	total: 681ms	remaining: 695ms
    495:	learn: 0.0777434	total: 682ms	remaining: 693ms
    496:	learn: 0.0771347	total: 684ms	remaining: 692ms
    497:	learn: 0.0770633	total: 685ms	remaining: 690ms
    498:	learn: 0.0766469	total: 686ms	remaining: 689ms
    499:	learn: 0.0764681	total: 687ms	remaining: 687ms
    500:	learn: 0.0760537	total: 688ms	remaining: 685ms
    501:	learn: 0.0759861	total: 689ms	remaining: 684ms
    502:	learn: 0.0759583	total: 690ms	remaining: 682ms
    503:	learn: 0.0755508	total: 691ms	remaining: 680ms
    504:	learn: 0.0755478	total: 692ms	remaining: 678ms
    505:	learn: 0.0751960	total: 693ms	remaining: 677ms
    506:	learn: 0.0747980	total: 694ms	remaining: 675ms
    507:	learn: 0.0747703	total: 695ms	remaining: 673ms
    508:	learn: 0.0746685	total: 696ms	remaining: 672ms
    509:	learn: 0.0743113	total: 697ms	remaining: 670ms
    510:	learn: 0.0742496	total: 698ms	remaining: 668ms
    511:	learn: 0.0738615	total: 699ms	remaining: 667ms
    512:	learn: 0.0737938	total: 700ms	remaining: 665ms
    513:	learn: 0.0737360	total: 701ms	remaining: 663ms
    514:	learn: 0.0736802	total: 702ms	remaining: 661ms
    515:	learn: 0.0736534	total: 703ms	remaining: 660ms
    516:	learn: 0.0736504	total: 704ms	remaining: 658ms
    517:	learn: 0.0736247	total: 705ms	remaining: 656ms
    518:	learn: 0.0732873	total: 706ms	remaining: 654ms
    519:	learn: 0.0729111	total: 707ms	remaining: 653ms
    520:	learn: 0.0729082	total: 708ms	remaining: 651ms
    521:	learn: 0.0729070	total: 709ms	remaining: 649ms
    522:	learn: 0.0729043	total: 710ms	remaining: 647ms
    523:	learn: 0.0728892	total: 711ms	remaining: 646ms
    524:	learn: 0.0728866	total: 711ms	remaining: 644ms
    525:	learn: 0.0728854	total: 712ms	remaining: 642ms
    526:	learn: 0.0728829	total: 713ms	remaining: 640ms
    527:	learn: 0.0725051	total: 714ms	remaining: 638ms
    528:	learn: 0.0725025	total: 715ms	remaining: 636ms
    529:	learn: 0.0721333	total: 716ms	remaining: 635ms
    530:	learn: 0.0720785	total: 717ms	remaining: 633ms
    531:	learn: 0.0720618	total: 717ms	remaining: 631ms
    532:	learn: 0.0717035	total: 719ms	remaining: 630ms
    533:	learn: 0.0716792	total: 720ms	remaining: 628ms
    534:	learn: 0.0716655	total: 721ms	remaining: 627ms
    535:	learn: 0.0716644	total: 722ms	remaining: 625ms
    536:	learn: 0.0715733	total: 723ms	remaining: 623ms
    537:	learn: 0.0715215	total: 724ms	remaining: 622ms
    538:	learn: 0.0714711	total: 725ms	remaining: 620ms
    539:	learn: 0.0714480	total: 726ms	remaining: 618ms
    540:	learn: 0.0714347	total: 727ms	remaining: 617ms
    541:	learn: 0.0714319	total: 728ms	remaining: 615ms
    542:	learn: 0.0711171	total: 729ms	remaining: 613ms
    543:	learn: 0.0707694	total: 730ms	remaining: 612ms
    544:	learn: 0.0707218	total: 732ms	remaining: 611ms
    545:	learn: 0.0703609	total: 733ms	remaining: 610ms
    546:	learn: 0.0703399	total: 735ms	remaining: 608ms
    547:	learn: 0.0702915	total: 736ms	remaining: 607ms
    548:	learn: 0.0698482	total: 737ms	remaining: 606ms
    549:	learn: 0.0698282	total: 738ms	remaining: 604ms
    550:	learn: 0.0694932	total: 739ms	remaining: 602ms
    551:	learn: 0.0694922	total: 740ms	remaining: 601ms
    552:	learn: 0.0694734	total: 741ms	remaining: 599ms
    553:	learn: 0.0691289	total: 742ms	remaining: 597ms
    554:	learn: 0.0687492	total: 743ms	remaining: 596ms
    555:	learn: 0.0682336	total: 745ms	remaining: 595ms
    556:	learn: 0.0682300	total: 746ms	remaining: 593ms
    557:	learn: 0.0681735	total: 747ms	remaining: 592ms
    558:	learn: 0.0681269	total: 749ms	remaining: 590ms
    559:	learn: 0.0681092	total: 750ms	remaining: 589ms
    560:	learn: 0.0677787	total: 751ms	remaining: 587ms
    561:	learn: 0.0677667	total: 751ms	remaining: 586ms
    562:	learn: 0.0677630	total: 753ms	remaining: 584ms
    563:	learn: 0.0677463	total: 754ms	remaining: 583ms
    564:	learn: 0.0677303	total: 755ms	remaining: 581ms
    565:	learn: 0.0674511	total: 756ms	remaining: 580ms
    566:	learn: 0.0671760	total: 757ms	remaining: 578ms
    567:	learn: 0.0671643	total: 758ms	remaining: 577ms
    568:	learn: 0.0668057	total: 760ms	remaining: 575ms
    569:	learn: 0.0667622	total: 760ms	remaining: 574ms
    570:	learn: 0.0665810	total: 762ms	remaining: 572ms
    571:	learn: 0.0664048	total: 763ms	remaining: 571ms
    572:	learn: 0.0660864	total: 764ms	remaining: 569ms
    573:	learn: 0.0657750	total: 765ms	remaining: 568ms
    574:	learn: 0.0657594	total: 766ms	remaining: 566ms
    575:	learn: 0.0657153	total: 767ms	remaining: 565ms
    576:	learn: 0.0654231	total: 768ms	remaining: 563ms
    577:	learn: 0.0654082	total: 770ms	remaining: 562ms
    578:	learn: 0.0653652	total: 772ms	remaining: 561ms
    579:	learn: 0.0650192	total: 773ms	remaining: 560ms
    580:	learn: 0.0647266	total: 775ms	remaining: 559ms
    581:	learn: 0.0646507	total: 776ms	remaining: 558ms
    582:	learn: 0.0643023	total: 777ms	remaining: 556ms
    583:	learn: 0.0642880	total: 778ms	remaining: 554ms
    584:	learn: 0.0639967	total: 780ms	remaining: 553ms
    585:	learn: 0.0636841	total: 781ms	remaining: 552ms
    586:	learn: 0.0636790	total: 782ms	remaining: 550ms
    587:	learn: 0.0636384	total: 783ms	remaining: 548ms
    588:	learn: 0.0636335	total: 783ms	remaining: 547ms
    589:	learn: 0.0636001	total: 784ms	remaining: 545ms
    590:	learn: 0.0633214	total: 786ms	remaining: 544ms
    591:	learn: 0.0633083	total: 787ms	remaining: 542ms
    592:	learn: 0.0630333	total: 789ms	remaining: 541ms
    593:	learn: 0.0630210	total: 790ms	remaining: 540ms
    594:	learn: 0.0629896	total: 791ms	remaining: 538ms
    595:	learn: 0.0629779	total: 791ms	remaining: 536ms
    596:	learn: 0.0629666	total: 792ms	remaining: 535ms
    597:	learn: 0.0629558	total: 793ms	remaining: 533ms
    598:	learn: 0.0627832	total: 794ms	remaining: 532ms
    599:	learn: 0.0627781	total: 795ms	remaining: 530ms
    600:	learn: 0.0626580	total: 796ms	remaining: 529ms
    601:	learn: 0.0623892	total: 798ms	remaining: 527ms
    602:	learn: 0.0623838	total: 798ms	remaining: 526ms
    603:	learn: 0.0621147	total: 799ms	remaining: 524ms
    604:	learn: 0.0620665	total: 801ms	remaining: 523ms
    605:	learn: 0.0618033	total: 802ms	remaining: 521ms
    606:	learn: 0.0615779	total: 803ms	remaining: 520ms
    607:	learn: 0.0611953	total: 804ms	remaining: 518ms
    608:	learn: 0.0611897	total: 805ms	remaining: 517ms
    609:	learn: 0.0611625	total: 806ms	remaining: 515ms
    610:	learn: 0.0608423	total: 807ms	remaining: 514ms
    611:	learn: 0.0607711	total: 808ms	remaining: 512ms
    612:	learn: 0.0607701	total: 809ms	remaining: 511ms
    613:	learn: 0.0607677	total: 809ms	remaining: 509ms
    614:	learn: 0.0605189	total: 810ms	remaining: 507ms
    615:	learn: 0.0605133	total: 811ms	remaining: 506ms
    616:	learn: 0.0602699	total: 812ms	remaining: 504ms
    617:	learn: 0.0602609	total: 813ms	remaining: 503ms
    618:	learn: 0.0599493	total: 815ms	remaining: 501ms
    619:	learn: 0.0596415	total: 815ms	remaining: 500ms
    620:	learn: 0.0594178	total: 816ms	remaining: 498ms
    621:	learn: 0.0594101	total: 817ms	remaining: 497ms
    622:	learn: 0.0593701	total: 818ms	remaining: 495ms
    623:	learn: 0.0591641	total: 819ms	remaining: 494ms
    624:	learn: 0.0591251	total: 820ms	remaining: 492ms
    625:	learn: 0.0588660	total: 822ms	remaining: 491ms
    626:	learn: 0.0588570	total: 823ms	remaining: 489ms
    627:	learn: 0.0586539	total: 824ms	remaining: 488ms
    628:	learn: 0.0584427	total: 825ms	remaining: 486ms
    629:	learn: 0.0583757	total: 826ms	remaining: 485ms
    630:	learn: 0.0583675	total: 827ms	remaining: 483ms
    631:	learn: 0.0583018	total: 828ms	remaining: 482ms
    632:	learn: 0.0582941	total: 829ms	remaining: 480ms
    633:	learn: 0.0582575	total: 830ms	remaining: 479ms
    634:	learn: 0.0579347	total: 831ms	remaining: 477ms
    635:	learn: 0.0579337	total: 831ms	remaining: 476ms
    636:	learn: 0.0577401	total: 832ms	remaining: 474ms
    637:	learn: 0.0577198	total: 833ms	remaining: 473ms
    638:	learn: 0.0577139	total: 834ms	remaining: 471ms
    639:	learn: 0.0574878	total: 836ms	remaining: 470ms
    640:	learn: 0.0574534	total: 836ms	remaining: 468ms
    641:	learn: 0.0572302	total: 838ms	remaining: 467ms
    642:	learn: 0.0572232	total: 839ms	remaining: 466ms
    643:	learn: 0.0570156	total: 840ms	remaining: 464ms
    644:	learn: 0.0570148	total: 840ms	remaining: 463ms
    645:	learn: 0.0569815	total: 841ms	remaining: 461ms
    646:	learn: 0.0569746	total: 842ms	remaining: 459ms
    647:	learn: 0.0569680	total: 843ms	remaining: 458ms
    648:	learn: 0.0566781	total: 844ms	remaining: 456ms
    649:	learn: 0.0566713	total: 845ms	remaining: 455ms
    650:	learn: 0.0566648	total: 846ms	remaining: 453ms
    651:	learn: 0.0566585	total: 847ms	remaining: 452ms
    652:	learn: 0.0564122	total: 848ms	remaining: 451ms
    653:	learn: 0.0564057	total: 849ms	remaining: 449ms
    654:	learn: 0.0561894	total: 850ms	remaining: 448ms
    655:	learn: 0.0559759	total: 851ms	remaining: 446ms
    656:	learn: 0.0557999	total: 853ms	remaining: 445ms
    657:	learn: 0.0557927	total: 853ms	remaining: 444ms
    658:	learn: 0.0554848	total: 855ms	remaining: 442ms
    659:	learn: 0.0554842	total: 855ms	remaining: 441ms
    660:	learn: 0.0552907	total: 857ms	remaining: 439ms
    661:	learn: 0.0549884	total: 858ms	remaining: 438ms
    662:	learn: 0.0547120	total: 859ms	remaining: 436ms
    663:	learn: 0.0545327	total: 860ms	remaining: 435ms
    664:	learn: 0.0543487	total: 861ms	remaining: 434ms
    665:	learn: 0.0543413	total: 862ms	remaining: 432ms
    666:	learn: 0.0543343	total: 863ms	remaining: 431ms
    667:	learn: 0.0541503	total: 864ms	remaining: 429ms
    668:	learn: 0.0541433	total: 865ms	remaining: 428ms
    669:	learn: 0.0539753	total: 866ms	remaining: 426ms
    670:	learn: 0.0539445	total: 867ms	remaining: 425ms
    671:	learn: 0.0537706	total: 868ms	remaining: 424ms
    672:	learn: 0.0536008	total: 869ms	remaining: 422ms
    673:	learn: 0.0535936	total: 870ms	remaining: 421ms
    674:	learn: 0.0534372	total: 871ms	remaining: 419ms
    675:	learn: 0.0532675	total: 872ms	remaining: 418ms
    676:	learn: 0.0529692	total: 873ms	remaining: 417ms
    677:	learn: 0.0529631	total: 874ms	remaining: 415ms
    678:	learn: 0.0529008	total: 876ms	remaining: 414ms
    679:	learn: 0.0527505	total: 877ms	remaining: 413ms
    680:	learn: 0.0525913	total: 878ms	remaining: 411ms
    681:	learn: 0.0522994	total: 879ms	remaining: 410ms
    682:	learn: 0.0522934	total: 880ms	remaining: 408ms
    683:	learn: 0.0521372	total: 881ms	remaining: 407ms
    684:	learn: 0.0519941	total: 882ms	remaining: 406ms
    685:	learn: 0.0518534	total: 883ms	remaining: 404ms
    686:	learn: 0.0518533	total: 884ms	remaining: 403ms
    687:	learn: 0.0518454	total: 885ms	remaining: 401ms
    688:	learn: 0.0518378	total: 885ms	remaining: 400ms
    689:	learn: 0.0516913	total: 887ms	remaining: 399ms
    690:	learn: 0.0513764	total: 889ms	remaining: 397ms
    691:	learn: 0.0513171	total: 890ms	remaining: 396ms
    692:	learn: 0.0512897	total: 891ms	remaining: 395ms
    693:	learn: 0.0512048	total: 892ms	remaining: 393ms
    694:	learn: 0.0510611	total: 893ms	remaining: 392ms
    695:	learn: 0.0509208	total: 894ms	remaining: 391ms
    696:	learn: 0.0507528	total: 896ms	remaining: 390ms
    697:	learn: 0.0507445	total: 897ms	remaining: 388ms
    698:	learn: 0.0506185	total: 898ms	remaining: 387ms
    699:	learn: 0.0506086	total: 899ms	remaining: 385ms
    700:	learn: 0.0504641	total: 900ms	remaining: 384ms
    701:	learn: 0.0504549	total: 904ms	remaining: 384ms
    702:	learn: 0.0504461	total: 905ms	remaining: 382ms
    703:	learn: 0.0503657	total: 907ms	remaining: 381ms
    704:	learn: 0.0503649	total: 908ms	remaining: 380ms
    705:	learn: 0.0503647	total: 909ms	remaining: 379ms
    706:	learn: 0.0502340	total: 911ms	remaining: 377ms
    707:	learn: 0.0502074	total: 912ms	remaining: 376ms
    708:	learn: 0.0501815	total: 913ms	remaining: 375ms
    709:	learn: 0.0500539	total: 914ms	remaining: 373ms
    710:	learn: 0.0499228	total: 916ms	remaining: 372ms
    711:	learn: 0.0497869	total: 917ms	remaining: 371ms
    712:	learn: 0.0497615	total: 918ms	remaining: 369ms
    713:	learn: 0.0497568	total: 919ms	remaining: 368ms
    714:	learn: 0.0497477	total: 919ms	remaining: 366ms
    715:	learn: 0.0496367	total: 920ms	remaining: 365ms
    716:	learn: 0.0496224	total: 922ms	remaining: 364ms
    717:	learn: 0.0496126	total: 924ms	remaining: 363ms
    718:	learn: 0.0494586	total: 925ms	remaining: 362ms
    719:	learn: 0.0493415	total: 926ms	remaining: 360ms
    720:	learn: 0.0493277	total: 927ms	remaining: 359ms
    721:	learn: 0.0493235	total: 928ms	remaining: 357ms
    722:	learn: 0.0492095	total: 929ms	remaining: 356ms
    723:	learn: 0.0490967	total: 931ms	remaining: 355ms
    724:	learn: 0.0490867	total: 931ms	remaining: 353ms
    725:	learn: 0.0490771	total: 932ms	remaining: 352ms
    726:	learn: 0.0489670	total: 933ms	remaining: 351ms
    727:	learn: 0.0489563	total: 934ms	remaining: 349ms
    728:	learn: 0.0489524	total: 936ms	remaining: 348ms
    729:	learn: 0.0489280	total: 936ms	remaining: 346ms
    730:	learn: 0.0486791	total: 938ms	remaining: 345ms
    731:	learn: 0.0486785	total: 939ms	remaining: 344ms
    732:	learn: 0.0485794	total: 940ms	remaining: 342ms
    733:	learn: 0.0485558	total: 941ms	remaining: 341ms
    734:	learn: 0.0482756	total: 942ms	remaining: 339ms
    735:	learn: 0.0482716	total: 943ms	remaining: 338ms
    736:	learn: 0.0481651	total: 944ms	remaining: 337ms
    737:	learn: 0.0480611	total: 945ms	remaining: 335ms
    738:	learn: 0.0480518	total: 946ms	remaining: 334ms
    739:	learn: 0.0477762	total: 947ms	remaining: 333ms
    740:	learn: 0.0475357	total: 948ms	remaining: 331ms
    741:	learn: 0.0472695	total: 949ms	remaining: 330ms
    742:	learn: 0.0470089	total: 951ms	remaining: 329ms
    743:	learn: 0.0469973	total: 952ms	remaining: 327ms
    744:	learn: 0.0469594	total: 953ms	remaining: 326ms
    745:	learn: 0.0468120	total: 954ms	remaining: 325ms
    746:	learn: 0.0466749	total: 955ms	remaining: 324ms
    747:	learn: 0.0465727	total: 957ms	remaining: 322ms
    748:	learn: 0.0464705	total: 958ms	remaining: 321ms
    749:	learn: 0.0464687	total: 959ms	remaining: 320ms
    750:	learn: 0.0463688	total: 960ms	remaining: 318ms
    751:	learn: 0.0463471	total: 961ms	remaining: 317ms
    752:	learn: 0.0463452	total: 962ms	remaining: 315ms
    753:	learn: 0.0462479	total: 963ms	remaining: 314ms
    754:	learn: 0.0461529	total: 964ms	remaining: 313ms
    755:	learn: 0.0460676	total: 966ms	remaining: 312ms
    756:	learn: 0.0459846	total: 967ms	remaining: 310ms
    757:	learn: 0.0458923	total: 968ms	remaining: 309ms
    758:	learn: 0.0456393	total: 969ms	remaining: 308ms
    759:	learn: 0.0453916	total: 970ms	remaining: 306ms
    760:	learn: 0.0453014	total: 971ms	remaining: 305ms
    761:	learn: 0.0452022	total: 972ms	remaining: 304ms
    762:	learn: 0.0451162	total: 974ms	remaining: 302ms
    763:	learn: 0.0450396	total: 975ms	remaining: 301ms
    764:	learn: 0.0450293	total: 975ms	remaining: 300ms
    765:	learn: 0.0450173	total: 977ms	remaining: 298ms
    766:	learn: 0.0447744	total: 978ms	remaining: 297ms
    767:	learn: 0.0447594	total: 979ms	remaining: 296ms
    768:	learn: 0.0446764	total: 980ms	remaining: 294ms
    769:	learn: 0.0446663	total: 981ms	remaining: 293ms
    770:	learn: 0.0445717	total: 982ms	remaining: 292ms
    771:	learn: 0.0445679	total: 983ms	remaining: 290ms
    772:	learn: 0.0445579	total: 984ms	remaining: 289ms
    773:	learn: 0.0445566	total: 985ms	remaining: 288ms
    774:	learn: 0.0445402	total: 986ms	remaining: 286ms
    775:	learn: 0.0445370	total: 987ms	remaining: 285ms
    776:	learn: 0.0444207	total: 988ms	remaining: 284ms
    777:	learn: 0.0444102	total: 989ms	remaining: 282ms
    778:	learn: 0.0443309	total: 990ms	remaining: 281ms
    779:	learn: 0.0443207	total: 991ms	remaining: 280ms
    780:	learn: 0.0443109	total: 992ms	remaining: 278ms
    781:	learn: 0.0440733	total: 993ms	remaining: 277ms
    782:	learn: 0.0439951	total: 995ms	remaining: 276ms
    783:	learn: 0.0437621	total: 996ms	remaining: 274ms
    784:	learn: 0.0437436	total: 997ms	remaining: 273ms
    785:	learn: 0.0437255	total: 998ms	remaining: 272ms
    786:	learn: 0.0437218	total: 999ms	remaining: 270ms
    787:	learn: 0.0437166	total: 1s	remaining: 269ms
    788:	learn: 0.0436466	total: 1s	remaining: 268ms
    789:	learn: 0.0436289	total: 1s	remaining: 266ms
    790:	learn: 0.0436116	total: 1s	remaining: 265ms
    791:	learn: 0.0435335	total: 1s	remaining: 264ms
    792:	learn: 0.0435289	total: 1s	remaining: 262ms
    793:	learn: 0.0434680	total: 1s	remaining: 261ms
    794:	learn: 0.0434648	total: 1.01s	remaining: 260ms
    795:	learn: 0.0432375	total: 1.01s	remaining: 258ms
    796:	learn: 0.0431633	total: 1.01s	remaining: 257ms
    797:	learn: 0.0430880	total: 1.01s	remaining: 256ms
    798:	learn: 0.0430788	total: 1.01s	remaining: 254ms
    799:	learn: 0.0429562	total: 1.01s	remaining: 253ms
    800:	learn: 0.0428846	total: 1.01s	remaining: 252ms
    801:	learn: 0.0428817	total: 1.01s	remaining: 251ms
    802:	learn: 0.0428119	total: 1.02s	remaining: 249ms
    803:	learn: 0.0425885	total: 1.02s	remaining: 248ms
    804:	learn: 0.0423736	total: 1.02s	remaining: 247ms
    805:	learn: 0.0423733	total: 1.02s	remaining: 245ms
    806:	learn: 0.0423042	total: 1.02s	remaining: 244ms
    807:	learn: 0.0422847	total: 1.02s	remaining: 243ms
    808:	learn: 0.0422818	total: 1.02s	remaining: 241ms
    809:	learn: 0.0422655	total: 1.02s	remaining: 240ms
    810:	learn: 0.0422627	total: 1.02s	remaining: 239ms
    811:	learn: 0.0420054	total: 1.02s	remaining: 237ms
    812:	learn: 0.0419863	total: 1.03s	remaining: 236ms
    813:	learn: 0.0419038	total: 1.03s	remaining: 235ms
    814:	learn: 0.0417058	total: 1.03s	remaining: 234ms
    815:	learn: 0.0416876	total: 1.03s	remaining: 232ms
    816:	learn: 0.0415858	total: 1.03s	remaining: 231ms
    817:	learn: 0.0415019	total: 1.03s	remaining: 230ms
    818:	learn: 0.0414853	total: 1.03s	remaining: 228ms
    819:	learn: 0.0414845	total: 1.03s	remaining: 227ms
    820:	learn: 0.0414838	total: 1.03s	remaining: 226ms
    821:	learn: 0.0412308	total: 1.04s	remaining: 224ms
    822:	learn: 0.0411727	total: 1.04s	remaining: 223ms
    823:	learn: 0.0411626	total: 1.04s	remaining: 222ms
    824:	learn: 0.0410999	total: 1.04s	remaining: 220ms
    825:	learn: 0.0410447	total: 1.04s	remaining: 219ms
    826:	learn: 0.0409793	total: 1.04s	remaining: 218ms
    827:	learn: 0.0409034	total: 1.04s	remaining: 217ms
    828:	learn: 0.0409028	total: 1.04s	remaining: 215ms
    829:	learn: 0.0408394	total: 1.04s	remaining: 214ms
    830:	learn: 0.0408232	total: 1.04s	remaining: 213ms
    831:	learn: 0.0408208	total: 1.05s	remaining: 211ms
    832:	learn: 0.0408157	total: 1.05s	remaining: 210ms
    833:	learn: 0.0406965	total: 1.05s	remaining: 209ms
    834:	learn: 0.0406339	total: 1.05s	remaining: 207ms
    835:	learn: 0.0406179	total: 1.05s	remaining: 206ms
    836:	learn: 0.0406078	total: 1.05s	remaining: 205ms
    837:	learn: 0.0405566	total: 1.05s	remaining: 203ms
    838:	learn: 0.0403430	total: 1.05s	remaining: 202ms
    839:	learn: 0.0402872	total: 1.05s	remaining: 201ms
    840:	learn: 0.0401691	total: 1.05s	remaining: 200ms
    841:	learn: 0.0400969	total: 1.06s	remaining: 198ms
    842:	learn: 0.0399805	total: 1.06s	remaining: 197ms
    843:	learn: 0.0399752	total: 1.06s	remaining: 196ms
    844:	learn: 0.0399653	total: 1.06s	remaining: 194ms
    845:	learn: 0.0399559	total: 1.06s	remaining: 193ms
    846:	learn: 0.0399015	total: 1.06s	remaining: 192ms
    847:	learn: 0.0398810	total: 1.06s	remaining: 190ms
    848:	learn: 0.0398277	total: 1.06s	remaining: 189ms
    849:	learn: 0.0397880	total: 1.06s	remaining: 188ms
    850:	learn: 0.0397727	total: 1.06s	remaining: 187ms
    851:	learn: 0.0397207	total: 1.07s	remaining: 185ms
    852:	learn: 0.0397109	total: 1.07s	remaining: 184ms
    853:	learn: 0.0396850	total: 1.07s	remaining: 183ms
    854:	learn: 0.0396797	total: 1.07s	remaining: 181ms
    855:	learn: 0.0396220	total: 1.07s	remaining: 180ms
    856:	learn: 0.0396071	total: 1.07s	remaining: 179ms
    857:	learn: 0.0394196	total: 1.07s	remaining: 178ms
    858:	learn: 0.0394051	total: 1.07s	remaining: 177ms
    859:	learn: 0.0393909	total: 1.08s	remaining: 175ms
    860:	learn: 0.0393406	total: 1.08s	remaining: 174ms
    861:	learn: 0.0392915	total: 1.08s	remaining: 173ms
    862:	learn: 0.0392437	total: 1.08s	remaining: 172ms
    863:	learn: 0.0392298	total: 1.08s	remaining: 170ms
    864:	learn: 0.0390207	total: 1.08s	remaining: 169ms
    865:	learn: 0.0390074	total: 1.08s	remaining: 168ms
    866:	learn: 0.0388783	total: 1.08s	remaining: 166ms
    867:	learn: 0.0388420	total: 1.08s	remaining: 165ms
    868:	learn: 0.0386072	total: 1.09s	remaining: 164ms
    869:	learn: 0.0385601	total: 1.09s	remaining: 163ms
    870:	learn: 0.0385142	total: 1.09s	remaining: 161ms
    871:	learn: 0.0385139	total: 1.09s	remaining: 160ms
    872:	learn: 0.0384919	total: 1.09s	remaining: 159ms
    873:	learn: 0.0382875	total: 1.09s	remaining: 158ms
    874:	learn: 0.0382872	total: 1.1s	remaining: 157ms
    875:	learn: 0.0380885	total: 1.1s	remaining: 155ms
    876:	learn: 0.0378941	total: 1.1s	remaining: 154ms
    877:	learn: 0.0378524	total: 1.1s	remaining: 153ms
    878:	learn: 0.0378521	total: 1.1s	remaining: 152ms
    879:	learn: 0.0378519	total: 1.1s	remaining: 150ms
    880:	learn: 0.0377997	total: 1.1s	remaining: 149ms
    881:	learn: 0.0377553	total: 1.1s	remaining: 148ms
    882:	learn: 0.0377164	total: 1.1s	remaining: 146ms
    883:	learn: 0.0377031	total: 1.1s	remaining: 145ms
    884:	learn: 0.0376218	total: 1.11s	remaining: 144ms
    885:	learn: 0.0376217	total: 1.11s	remaining: 143ms
    886:	learn: 0.0376215	total: 1.11s	remaining: 141ms
    887:	learn: 0.0376083	total: 1.11s	remaining: 140ms
    888:	learn: 0.0375666	total: 1.11s	remaining: 139ms
    889:	learn: 0.0373763	total: 1.11s	remaining: 137ms
    890:	learn: 0.0370848	total: 1.11s	remaining: 136ms
    891:	learn: 0.0370723	total: 1.11s	remaining: 135ms
    892:	learn: 0.0370317	total: 1.11s	remaining: 134ms
    893:	learn: 0.0369713	total: 1.12s	remaining: 132ms
    894:	learn: 0.0369712	total: 1.12s	remaining: 131ms
    895:	learn: 0.0367867	total: 1.12s	remaining: 130ms
    896:	learn: 0.0367332	total: 1.12s	remaining: 129ms
    897:	learn: 0.0367212	total: 1.12s	remaining: 127ms
    898:	learn: 0.0364983	total: 1.12s	remaining: 126ms
    899:	learn: 0.0364899	total: 1.12s	remaining: 125ms
    900:	learn: 0.0364898	total: 1.12s	remaining: 124ms
    901:	learn: 0.0364128	total: 1.13s	remaining: 122ms
    902:	learn: 0.0362331	total: 1.13s	remaining: 121ms
    903:	learn: 0.0360298	total: 1.13s	remaining: 120ms
    904:	learn: 0.0358289	total: 1.13s	remaining: 119ms
    905:	learn: 0.0358173	total: 1.13s	remaining: 117ms
    906:	learn: 0.0358055	total: 1.13s	remaining: 116ms
    907:	learn: 0.0357653	total: 1.13s	remaining: 115ms
    908:	learn: 0.0357630	total: 1.13s	remaining: 114ms
    909:	learn: 0.0357629	total: 1.14s	remaining: 112ms
    910:	learn: 0.0355457	total: 1.14s	remaining: 111ms
    911:	learn: 0.0355105	total: 1.14s	remaining: 110ms
    912:	learn: 0.0352340	total: 1.14s	remaining: 109ms
    913:	learn: 0.0350613	total: 1.14s	remaining: 107ms
    914:	learn: 0.0350612	total: 1.14s	remaining: 106ms
    915:	learn: 0.0350167	total: 1.14s	remaining: 105ms
    916:	learn: 0.0350053	total: 1.14s	remaining: 104ms
    917:	learn: 0.0349969	total: 1.14s	remaining: 102ms
    918:	learn: 0.0348867	total: 1.15s	remaining: 101ms
    919:	learn: 0.0348307	total: 1.15s	remaining: 99.8ms
    920:	learn: 0.0345695	total: 1.15s	remaining: 98.6ms
    921:	learn: 0.0345582	total: 1.15s	remaining: 97.4ms
    922:	learn: 0.0345503	total: 1.16s	remaining: 96.6ms
    923:	learn: 0.0345502	total: 1.16s	remaining: 95.4ms
    924:	learn: 0.0345257	total: 1.16s	remaining: 94.2ms
    925:	learn: 0.0344889	total: 1.16s	remaining: 93.1ms
    926:	learn: 0.0344666	total: 1.17s	remaining: 91.8ms
    927:	learn: 0.0342081	total: 1.17s	remaining: 90.6ms
    928:	learn: 0.0341723	total: 1.17s	remaining: 89.4ms
    929:	learn: 0.0341374	total: 1.17s	remaining: 88.1ms
    930:	learn: 0.0340962	total: 1.17s	remaining: 86.9ms
    931:	learn: 0.0338308	total: 1.18s	remaining: 85.8ms
    932:	learn: 0.0338307	total: 1.18s	remaining: 84.5ms
    933:	learn: 0.0338009	total: 1.18s	remaining: 83.3ms
    934:	learn: 0.0337611	total: 1.18s	remaining: 82ms
    935:	learn: 0.0337326	total: 1.18s	remaining: 80.8ms
    936:	learn: 0.0337254	total: 1.18s	remaining: 79.5ms
    937:	learn: 0.0337185	total: 1.18s	remaining: 78.3ms
    938:	learn: 0.0337066	total: 1.19s	remaining: 77ms
    939:	learn: 0.0336681	total: 1.19s	remaining: 75.8ms
    940:	learn: 0.0336680	total: 1.19s	remaining: 74.5ms
    941:	learn: 0.0334060	total: 1.19s	remaining: 73.3ms
    942:	learn: 0.0332040	total: 1.19s	remaining: 72ms
    943:	learn: 0.0329456	total: 1.19s	remaining: 70.8ms
    944:	learn: 0.0328944	total: 1.2s	remaining: 69.6ms
    945:	learn: 0.0328640	total: 1.2s	remaining: 68.3ms
    946:	learn: 0.0328376	total: 1.2s	remaining: 67ms
    947:	learn: 0.0328260	total: 1.2s	remaining: 65.8ms
    948:	learn: 0.0328163	total: 1.2s	remaining: 64.5ms
    949:	learn: 0.0326267	total: 1.2s	remaining: 63.3ms
    950:	learn: 0.0326173	total: 1.2s	remaining: 62.1ms
    951:	learn: 0.0323636	total: 1.21s	remaining: 60.8ms
    952:	learn: 0.0323277	total: 1.21s	remaining: 59.6ms
    953:	learn: 0.0323277	total: 1.21s	remaining: 58.3ms
    954:	learn: 0.0323095	total: 1.21s	remaining: 57.1ms
    955:	learn: 0.0323095	total: 1.21s	remaining: 55.8ms
    956:	learn: 0.0322980	total: 1.21s	remaining: 54.6ms
    957:	learn: 0.0322507	total: 1.22s	remaining: 53.3ms
    958:	learn: 0.0322221	total: 1.22s	remaining: 52.1ms
    959:	learn: 0.0321973	total: 1.22s	remaining: 50.8ms
    960:	learn: 0.0321793	total: 1.22s	remaining: 49.6ms
    961:	learn: 0.0321520	total: 1.22s	remaining: 48.3ms
    962:	learn: 0.0321284	total: 1.22s	remaining: 47.1ms
    963:	learn: 0.0320949	total: 1.23s	remaining: 45.8ms
    964:	learn: 0.0320832	total: 1.23s	remaining: 44.6ms
    965:	learn: 0.0320743	total: 1.23s	remaining: 43.3ms
    966:	learn: 0.0320486	total: 1.23s	remaining: 42ms
    967:	learn: 0.0320370	total: 1.23s	remaining: 40.8ms
    968:	learn: 0.0318733	total: 1.23s	remaining: 39.5ms
    969:	learn: 0.0318622	total: 1.24s	remaining: 38.2ms
    970:	learn: 0.0318410	total: 1.24s	remaining: 37ms
    971:	learn: 0.0315939	total: 1.24s	remaining: 35.7ms
    972:	learn: 0.0315617	total: 1.24s	remaining: 34.4ms
    973:	learn: 0.0315367	total: 1.24s	remaining: 33.2ms
    974:	learn: 0.0315305	total: 1.24s	remaining: 31.9ms
    975:	learn: 0.0313708	total: 1.25s	remaining: 30.7ms
    976:	learn: 0.0313708	total: 1.25s	remaining: 29.5ms
    977:	learn: 0.0312133	total: 1.25s	remaining: 28.2ms
    978:	learn: 0.0310321	total: 1.26s	remaining: 26.9ms
    979:	learn: 0.0308782	total: 1.26s	remaining: 25.7ms
    980:	learn: 0.0308350	total: 1.26s	remaining: 24.4ms
    981:	learn: 0.0308145	total: 1.26s	remaining: 23.1ms
    982:	learn: 0.0308145	total: 1.27s	remaining: 21.9ms
    983:	learn: 0.0307895	total: 1.27s	remaining: 20.7ms
    984:	learn: 0.0307829	total: 1.28s	remaining: 19.5ms
    985:	learn: 0.0307718	total: 1.28s	remaining: 18.2ms
    986:	learn: 0.0307705	total: 1.28s	remaining: 16.9ms
    987:	learn: 0.0307705	total: 1.28s	remaining: 15.6ms
    988:	learn: 0.0307461	total: 1.28s	remaining: 14.3ms
    989:	learn: 0.0305362	total: 1.29s	remaining: 13ms
    990:	learn: 0.0305126	total: 1.29s	remaining: 11.7ms
    991:	learn: 0.0304895	total: 1.29s	remaining: 10.4ms
    992:	learn: 0.0304787	total: 1.29s	remaining: 9.1ms
    993:	learn: 0.0304786	total: 1.29s	remaining: 7.8ms
    994:	learn: 0.0304681	total: 1.29s	remaining: 6.5ms
    995:	learn: 0.0304456	total: 1.29s	remaining: 5.2ms
    996:	learn: 0.0304456	total: 1.3s	remaining: 3.9ms
    997:	learn: 0.0304456	total: 1.3s	remaining: 2.6ms
    998:	learn: 0.0304386	total: 1.3s	remaining: 1.3ms
    999:	learn: 0.0302030	total: 1.3s	remaining: 0us
    trained y [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    predict y [0.01295236 0.00398439 0.97056257 0.92993422 0.99008662 0.01035077
     0.98927101 0.02936561 0.97276265 1.05706333 0.97905977 1.01208608
     1.0288173  0.01468734]


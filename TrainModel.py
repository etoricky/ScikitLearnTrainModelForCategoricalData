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

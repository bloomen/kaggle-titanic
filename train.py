import pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.preprocessing.data import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.utils import assert_all_finite


class CopyTransformer(TransformerMixin):

    def transform(self, df, *_):
        df = pd.DataFrame(df, copy=True)
        return df

    def fit(self, *_):
        return self


class FloatConverter(TransformerMixin):

    def transform(self, df, *_):
        df = pd.DataFrame(df, copy=True, dtype=float)
        return df

    def fit(self, *_):
        return self


class MedianImputer(TransformerMixin):

    def __init__(self, column):
        self._column = column
        self._imputer = SimpleImputer(strategy='median')

    def transform(self, df, *_):
        df[self._column] = self._imputer.transform(pd.DataFrame(df[self._column]))[:, 0]
        return df

    def fit(self, df, *_):
        self._imputer.fit(pd.DataFrame(df[self._column]))
        return self


class ConstantImputer(TransformerMixin):

    def __init__(self, column, fill_value):
        self._column = column
        self._imputer = SimpleImputer(strategy='constant', fill_value=fill_value)

    def transform(self, df, *_):
        df[self._column] = self._imputer.transform(pd.DataFrame(df[self._column]))[:, 0]
        return df

    def fit(self, df, *_):
        self._imputer.fit(pd.DataFrame(df[self._column]))
        return self


class CheckModel:
    
    def fit(self, X, *_):
        assert_all_finite(df)
        print('CheckModel Start -------------')
        print(X.head())
        print(X.describe())
        print(X.dtypes)
        print(X.isna().sum())
        print('CheckModel End -------------')

    def predict(self, df, *_):
        pass


class OneHotEncoder(TransformerMixin):

    def __init__(self, column):
        self._column = column
        self._encoder = preprocessing.OneHotEncoder(sparse=False)

    def transform(self, df, *_):
        onehot = self._encoder.transform(pd.DataFrame(df[self._column]))
        df_enc = pd.DataFrame(onehot, columns=[self._column + '_' + c for c in self._encoder.get_feature_names()])
        df_enc.index = df.index
        df = pd.concat([df, df_enc], axis=1, verify_integrity=True)
        return df.drop([self._column], axis=1)

    def fit(self, df, *_):
        self._encoder.fit(pd.DataFrame(df[self._column]))
        return self


class LabelEncoder(TransformerMixin):

    def __init__(self, column):
        self._column = column
        self._encoder = label.LabelEncoder()

    def transform(self, df, *_):
        df[self._column] = self._encoder.transform(df[self._column])
        return df

    def fit(self, df, *_):
        self._encoder.fit(df[self._column])
        return self


class PolyFeatureGenerator(TransformerMixin):

    def __init__(self, degree):
        self._poly = PolynomialFeatures(degree=degree)

    def transform(self, df, *_):
        df_poly = self._poly.transform(df)
        df_poly = pd.DataFrame(df_poly, columns=self._poly.get_feature_names())
        df_poly.index = df.index
        return pd.concat([df, df_poly], axis=1)

    def fit(self, df, *_):
        self._poly.fit(df)
        return self


def hist_all(df):
    for col in df:
        plt.figure()
        df[col].plot.hist(bins=20, title=col)


def corr_all(df):
    corr = df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                cmap=sns.diverging_palette(220, 10, as_cmap=True))


def main():
    df = pd.read_csv('train.csv')
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # print(df.head())
    # print(df.isna().sum())

    y = df['Survived']
    X = df.drop(['Survived'], axis=1)

    model = Pipeline([
        ('copy_transformer', CopyTransformer()),
        ('age_imputer', MedianImputer('Age')),
        ('embarked_imputer', ConstantImputer('Embarked', 'Q')),
        ('embarked_encoder', OneHotEncoder('Embarked')),
        ('sex_encoder', LabelEncoder('Sex')),
        ('float_converter', FloatConverter()),
        ('poly', PolyFeatureGenerator(2)),
        ('pca', PCA(n_components=20)),
        ('scaler', MinMaxScaler(feature_range=(-1, 1))),
        ('model', SVC(gamma='scale')),
#        ('model', CheckModel()),
    ])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # 
    # assert_all_finite(X_train)
    # model.fit(X_train, y_train)
    # y_predict = model.predict(X_test)
    # 
    # print(accuracy_score(y_test, y_predict))

    scores = cross_validate(model, X, y, scoring='accuracy', cv=3)  # be aware of the accuracy paradox
    print(scores['test_score'])
    print(scores['test_score'].mean())

    # X = model.fit_transform(X)
    # print('-------------')
    # print(X.head())
    # print(X.isna().sum())
    # print(X.describe())
    # print(X.dtypes)
    # 
    # assert_all_finite(X)
    
    # hist_all(X)
    # plt.show()


if __name__ == '__main__':
    main()

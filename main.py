import pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing.data import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing.data import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.utils import assert_all_finite
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import argparse
import pickle
np.random.seed(42)


class CopyTransformer(TransformerMixin):

    def transform(self, df, *_):
        assert_all_finite(df)
        df = pd.DataFrame(df, copy=True)
        assert_all_finite(df)
        return df

    def fit(self, *_):
        return self


class FloatConverter(TransformerMixin):

    def transform(self, df, *_):
        assert_all_finite(df)
        df = pd.DataFrame(df, copy=True, dtype=float)
        assert_all_finite(df)
        return df

    def fit(self, *_):
        return self


class MedianImputer(TransformerMixin):

    def __init__(self, column):
        self._column = column
        self._imputer = SimpleImputer(strategy='median')

    def transform(self, df, *_):
        assert_all_finite(df)
        df[self._column] = self._imputer.transform(pd.DataFrame(df[self._column]))[:, 0]
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        self._imputer.fit(pd.DataFrame(df[self._column]))
        return self


class ConstantImputer(TransformerMixin):

    def __init__(self, column, fill_value):
        self._column = column
        self._imputer = SimpleImputer(strategy='constant', fill_value=fill_value)

    def transform(self, df, *_):
        assert_all_finite(df)
        df[self._column] = self._imputer.transform(pd.DataFrame(df[self._column]))[:, 0]
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        self._imputer.fit(pd.DataFrame(df[self._column]))
        return self


class NoiseImputer(TransformerMixin):

    def __init__(self, column):
        self._column = column
        self._min = None
        self._max = None

    def transform(self, df, *_):
        assert_all_finite(df)
        df[self._column] = df[self._column].apply(self._random_value)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        self._min = df[self._column].min()
        self._max = df[self._column].max()
        return self
        
    def _random_value(self, x):
        if np.isnan(x):
            return np.random.uniform(self._min, self._max)
        else:
            return x


class OneHotEncoder(TransformerMixin):

    def __init__(self, column):
        self._column = column
        self._encoder = preprocessing.OneHotEncoder(sparse=False)

    def transform(self, df, *_):
        assert_all_finite(df)
        onehot = self._encoder.transform(pd.DataFrame(df[self._column]))
        df_enc = pd.DataFrame(onehot, columns=[self._column + '_' + c for c in self._encoder.get_feature_names()])
        df_enc.index = df.index
        df = pd.concat([df, df_enc], axis=1, verify_integrity=True)
        df = df.drop([self._column], axis=1)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        self._encoder.fit(pd.DataFrame(df[self._column]))
        return self


class LabelEncoder(TransformerMixin):

    def __init__(self, column):
        self._column = column
        self._encoder = label.LabelEncoder()

    def transform(self, df, *_):
        assert_all_finite(df)
        df[self._column] = self._encoder.transform(df[self._column])
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        self._encoder.fit(df[self._column])
        return self


class PolyFeatureGenerator(TransformerMixin):

    def __init__(self, degree):
        self._poly = PolynomialFeatures(degree=degree)

    def transform(self, df, *_):
        assert_all_finite(df)
        df_poly = self._poly.transform(df)
        df_poly = pd.DataFrame(df_poly, columns=self._poly.get_feature_names())
        df_poly.index = df.index
        df = pd.concat([df, df_poly], axis=1)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        self._poly.fit(df)
        return self


class DiffFeatureGenerator(TransformerMixin):

    def transform(self, df, *_):
        assert_all_finite(df)
        diff = {}
        for c0 in df:
            for c1 in df:
                if c0 != c1:
                    diff['{}-{}'.format(c0, c1)] = df[c0] - df[c1]
        df_diff = pd.DataFrame(diff)
        df_diff.index = df.index
        df = pd.concat([df, df_diff], axis=1)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        return self


class FractionFeatureGenerator(TransformerMixin):

    def transform(self, df, *_):
        assert_all_finite(df)
        fraction = {}
        for c0 in df:
            for c1 in df:
                if c0 != c1:
                    fraction['{}/{}'.format(c0, c1)] = df[c0] / df[c1].replace(0, 1)
        df_fraction = pd.DataFrame(fraction)
        df_fraction.index = df.index
        df = pd.concat([df, df_fraction], axis=1)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        return self


class InteractionFeatureGenerator(TransformerMixin):

    def transform(self, df, *_):
        assert_all_finite(df)
        interaction = {}
        for c0 in df:
            for c1 in df:
                if c0 != c1:
                    interaction['{}-{}'.format(c0, c1)] = df[c0] - df[c1]
                    interaction['{}*{}'.format(c0, c1)] = df[c0] * df[c1]
#                    interaction['{}/{}'.format(c0, c1)] = df[c0] / df[c1].replace(0, 1)
        df_interaction = pd.DataFrame(interaction)
        df_interaction.index = df.index
        df = pd.concat([df, df_interaction], axis=1)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        return self


class ApplyFunctor(TransformerMixin):

    def __init__(self, column, functor):
        self._column = column
        self._functor = functor

    def transform(self, df, *_):
        assert_all_finite(df)
        df[self._column] = df[self._column].apply(self._functor)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        return self


class ColumnDropper(TransformerMixin):

    def __init__(self, columns):
        self._columns = columns

    def transform(self, df, *_):
        assert_all_finite(df)
        df = df.drop(self._columns, axis=1)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        return self


class Scaler(TransformerMixin):

    def __init__(self):
        self._scaler = MinMaxScaler(feature_range=(-1, 1))

    def transform(self, df, *_):
        assert_all_finite(df)
        scaled = self._scaler.transform(df)
        df = pd.DataFrame(scaled, columns=df.columns)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        self._scaler.fit(df)
        return self


class CabinTransformer(TransformerMixin):

    def transform(self, df, *_):
        assert_all_finite(df)
        decks = []
        numbers = []
        for val in df['Cabin']:
            num = np.nan
            val = str(val)
            if val != 'nan':
                decks.append(ord(val[0]))
                if len(val) > 1:
                    try:
                        num = int(val.split()[0][1:])
                    except:
                        pass
            else:
                decks.append(np.nan)
            numbers.append(num)
        df_cabin = pd.DataFrame(dict(cabin_deck=decks, cabin_number=numbers))
        df_cabin.index = df.index
        df = pd.concat([df, df_cabin], axis=1)
        df = df.drop(['Cabin'], axis=1)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        return self


class TicketTransformer(TransformerMixin):

    def transform(self, df, *_):
        assert_all_finite(df)
        numbers = []
        for val in df['Ticket']:
            num = np.nan
            for v in reversed(val.split()):
                try:
                    num = int(v)
                    break
                except:
                    pass
            numbers.append(num)
        df_ticket = pd.DataFrame(dict(ticket_number=numbers))
        df_ticket.index = df.index
        df = pd.concat([df, df_ticket], axis=1)
        df = df.drop(['Ticket'], axis=1)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        return self


class NameTransformer(TransformerMixin):

    def __init__(self):
        # male = 0, female = 1
        self._titles = {
            'Don.': 0, 
            'Master.': 0,
            'Rev.': 0, 
            'Major.': 0, 
            'Sir.': 0, 
            'Ms.': 1, 
            'Mlle.': 1,
            'Col.': 1, 
            'Jonkheer.': 0,
            'Miss.': 1, 
            'Mr.': 1, 
            'Mme.': 1, 
            'Mrs.': 1, 
            'Countess.': 1,
            'Dr.': 0, 
            'Capt.': 0,
            'Lady.': 1,
            'Dona.': 1,
        }

    def transform(self, df, *_):
        assert_all_finite(df)
        gender = []
        for val in df['Name']:
            title = 0
            for v in val.split():
                if v[-1] == '.':
                    title = self._titles.get(v, 1)
                    break
            gender.append(title)
        df_name = pd.DataFrame(dict(name_gender=gender))
        df_name.index = df.index
        df = pd.concat([df, df_name], axis=1)
        df = df.drop(['Name'], axis=1)
        assert_all_finite(df)
        return df

    def fit(self, df, *_):
        return self


def hist_all(df):
    for col in df:
        plt.figure(col)
        df[col].plot.hist(bins=20, title=col)


def corr_all(df):
    plt.figure('Correlation')
    corr = df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                cmap=sns.diverging_palette(220, 10, as_cmap=True))


def bounded_log(x):
    return np.log(x) if x > 0 else 1.7


def base_pipeline():
    return [
        ('copy_transformer', CopyTransformer()),
        ('fare_imputer', NoiseImputer('Fare')),
        ('fare_log', ApplyFunctor('Fare', bounded_log)),
        ('age_imputer', NoiseImputer('Age')),
        ('name_transformer', NameTransformer()),
        ('ticket_transformer', TicketTransformer()),
        ('ticket_number_imputer', NoiseImputer('ticket_number')),
        ('ticker_number_log', ApplyFunctor('ticket_number', np.log)),
        ('cabin_transformer', CabinTransformer()),
        ('cabin_deck_imputer', NoiseImputer('cabin_deck')),
        ('cabin_number_imputer', NoiseImputer('cabin_number')),
        ('embarked_imputer', ConstantImputer('Embarked', 'Q')),
        ('embarked_encoder', OneHotEncoder('Embarked')),
        ('sex_imputer', ConstantImputer('Sex', 'male')),
        ('sex_encoder', LabelEncoder('Sex')),
        ('float_converter', FloatConverter()),
        ('scaler1', Scaler()),
    ]


def train_pipeline():
    pl = base_pipeline()
    pl.extend([
        ('dropper', ColumnDropper(['cabin_deck', 'cabin_number', 'ticket_number'])),
        ('interaction', InteractionFeatureGenerator()),
        ('scaler2', Scaler()),
        ('pca', PCA(n_components=50)),
#        ('model', MLPClassifier((128,), max_iter=200, learning_rate='constant', learning_rate_init=0.001, verbose=True))
#        ('model', RandomForestClassifier(n_estimators=100, max_depth=10)),
#        ('model', SVC(gamma=1e-3, C=1e1, kernel='rbf', probability=True)),
        ('model', SVC(gamma=1e-3, C=1e2, kernel='rbf', probability=True)),
    ])
    return Pipeline(pl)


def train_data():
    df = pd.read_csv('input/train.csv')
    df = df.drop(['PassengerId'], axis=1)
    y = df['Survived']
    X = df.drop(['Survived'], axis=1)
    return X, y


def test_data():
    df = pd.read_csv('input/test.csv')
    ids = df['PassengerId']
    X = df.drop(['PassengerId'], axis=1)
    return X, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", help="Ensure good data quality",
                        action="store_true")
    parser.add_argument("--train", help="Train the model", type=str,
                        choices=['cv', 'best'])
    parser.add_argument("--evaluate", help="Evaluate the model", type=str,
                        choices=['cv', 'best'])
    parser.add_argument("--submission", help="Generate submission on test data", type=str,
                        choices=['cv', 'best'])
    args = parser.parse_args()

    try:
        os.mkdir('output')
    except FileExistsError:
        pass

    if args.quality:
        X, y = train_data()
        model = Pipeline(base_pipeline())
        X = model.fit_transform(X)
        X = pd.concat([X, y], axis=1)
        print('X.dtypes Start -------------')
        print(X.dtypes)
        print('X.dtypes End -------------')
        print('X.head() Start -------------')
        print(X.head())
        print('X.head() End -------------')
        print('X.isna().sum() Start -------------')
        print(X.isna().sum())
        print('X.isna().sum() End -------------')
        print('X.describe() Start -------------')
        print(X.describe())
        print('X.describe() End -------------')
        assert_all_finite(X)
        hist_all(X)
        corr_all(X)
        plt.show(block=False)
        input("Press [enter] to continue.")
        return

    if args.train == 'cv':
        X, y = train_data()
        model = train_pipeline()
        scores = cross_validate(model, X, y, scoring='accuracy', cv=10)  # be aware of the accuracy paradox
        print('scores =', scores['test_score'])
        print('mean score =', scores['test_score'].mean())
        print('std score =', scores['test_score'].std())
        model = train_pipeline()
        model.fit(X, y)
        with open('output/modelcv.pickle', 'wb') as f:
            pickle.dump(model, f)
        return

    if args.train == 'best':
        X, y = train_data()
        model = train_pipeline()
        # SVC
        params = {
            'model__gamma': [1e-5, 1e-4, 1e-3, 1e-2],
            'model__C': [1e1, 1e2, 1e3, 1e4, 1e5],
        }
        # MLP
        # params = {
        #     'model__hidden_layer_sizes': range(120, 180, 10),
        #     'model__learning_rate_init': [1e-3, 1e-2, 1e-1],
        # }
        gridcv = GridSearchCV(model, params, verbose=2, cv=5, scoring='accuracy')
        gridcv.fit(X, y)
        print('best score =', gridcv.best_score_)
        print('best params =', gridcv.best_params_)
        best = gridcv.best_estimator_
        best.fit(X, y)
        with open('output/modelbest.pickle', 'wb') as f:
            pickle.dump(best, f)
        return

    if args.submission:
        threshold = 0.655 # from ROC curve
        X, ids = test_data()
        with open('output/model{}.pickle'.format(args.submission), 'rb') as f:
            model = pickle.load(f)
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = [0 if x <= threshold else 1 for x in y_prob]
        y_pred = pd.DataFrame(y_pred, columns=['Survived'])
        y_pred.index = ids.index
        submission = pd.concat([ids,y_pred], axis=1)
        submission.to_csv('output/submission{}.csv'.format(args.submission), index=False)
        return

    if args.evaluate:
        X, y = train_data()
        with open('output/model{}.pickle'.format(args.evaluate), 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y.values, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure("ROC")
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show(block=False)
        input("Press [enter] to continue.")
        return


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


class RespHelpers:
    """RespHelpers class.

    This is simply a bit of house cleaning: this helper class houses all the functions
    being used to evaluate the extracted features. See evaluation.py for results and plots.

    """

    @staticmethod
    def column_to_01(column, threshold=0.5):
        return np.ceil(column - threshold).astype('int')

    @staticmethod
    def get_labelwise_acc(y_true, y_pred):
        acc_ = []
        for i in range(y_true.shape[1]):
            acc_.append(sklearn.metrics.accuracy_score(y_true[:, i], y_pred[:, i]))
        return acc_

    @staticmethod
    def Xy_splitter(data, n_labels=1):
        try:
            data = data.to_numpy()
        except TypeError:
            print('Accepts pandas DataFrame type.')

        X = data[:, :-n_labels]
        y = data[:, -n_labels:]
        return X, y

    def get_PCA(self, feature_data, n_components, n_labels=2):
        X, y = self.Xy_splitter(feature_data, n_labels=n_labels)
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X = pca.transform(X)
        return pd.DataFrame(np.hstack((X, y)))

    def cross_validate(self, kfold_dataset, clf, n_splits=5, n_labels=1, verbose=True):
        loss = []
        acc = []
        prec = []
        recall = []
        f1 = []
        support = []

        kf = KFold(n_splits=n_splits, shuffle=True)

        for i, (train_index, validate_index) in enumerate(kf.split(kfold_dataset)):
            train_data = kfold_dataset.iloc[train_index, :]
            val_data = kfold_dataset.iloc[validate_index, :]

            train_X, train_y = self.Xy_splitter(train_data, n_labels=n_labels)
            val_X, val_y = self.Xy_splitter(val_data, n_labels=n_labels)

            print("Training fold %d..." % (i + 1))
            clf.fit(train_X, train_y)
            y_pred = clf.predict(val_X)

            loss.append(sklearn.metrics.hamming_loss(val_y, y_pred))
            acc.append(self.get_labelwise_acc(val_y, y_pred))
            p_, r_, f1_, s_ = sklearn.metrics.precision_recall_fscore_support(val_y, y_pred)
            prec.append(p_)
            recall.append(r_)
            f1.append(f1_)
            support.append(s_)

        loss_ = np.asarray(loss).mean()
        acc_ = np.asarray(acc).mean(axis=0)
        prec_ = np.asarray(prec).mean(axis=0)
        recall_ = np.asarray(recall).mean(axis=0)
        f1_ = np.asarray(f1).mean(axis=0)
        support_ = np.asarray(support).mean(axis=0)

        if verbose:
            print('Cross validation mean values -----')
            print("Hamming loss: %f" % loss_)
            print("Crackle acc: %f\tWheeze acc: %f" % (acc_[0], acc_[1]))
            print("Crackle prec: %f\tWheeze prec: %f" % (prec_[0], prec_[1]))
            print("Crackle recall: %f\tWheeze recall: %f" % (recall_[0], recall_[1]))
            print("Crackle f1: %f\tWheeze f1: %f" % (f1_[0], f1_[1]))
            print("Crackle support: %f\tWheeze support: %f" % (support_[0], support_[1]))
        return loss_, acc_, prec_, recall_, f1_, support_

    def get_pca_training_stats(self, feature_data, low, high, step, clf, verbose=True):
        loss = []
        acc = []
        prec = []
        recall = []
        f1 = []

        for i in range(low, high, step):
            print('\nNum PCA components = %d\n' % i)
            pca_data = self.get_PCA(feature_data, n_labels=2, n_components=i)
            loss_, acc_, prec_, recall_, f1_, support_ = self.cross_validate(pca_data, clf, n_splits=5, n_labels=2,
                                                                             verbose=verbose)

            loss.append(loss_)
            acc.append(acc_)
            prec.append(prec_)
            recall.append(recall_)
            f1.append(f1_)
        return loss, acc, prec, recall, f1, support_
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from helper_functions import RespHelpers
from resp_extraction import feature_data

""" Evaluation of SVM classification models trained on FeatureExtractor features.

    This script runs two experiments as outlined in the final project paper. In each case
    wheeze and crackle detection are treated as independent tasks (i.e., multi-label paradigm).
    Two bar graphs are presented: the first representing crackle detection results, the second
    for wheeze detection results. Bar graphs do not include titles to facilitate formatting for
    the final paper in which they are copied.
    
    EXPERIMENT 1:
    Train an SVM on subsets of the total feature set. Here in three versions:
    (1) MFCCs only.
    (2) YAMNet embeds, frame-wise means only.
    (3) YAMNet embeds, frame-wise means and standard deviations (a.k.a. 'YAMNet all').
    
    EXPERIMENT 2:
    Train an SVM on various PCA compressions of the total feature set. Starting with 10
    principle components, the the test iterates up to 20 principle components.

    EXAMPLE USE from TERMINAL:
    
    python3 evaluation.py 1 1           --- runs both experiments and plot results as bar graphs.
    python3 evaluation.py 1             --- runs first experiment only.
    python3 evaluation.py 0 1           --- runs second experiment only.

"""


def get_subsection_with_labels(data, n_labels, from_, to_):
    labels = data.iloc[:, -n_labels:]
    subsection = data.iloc[:, from_:to_]
    return subsection.join(labels)


def bar_graph(accuracies, precisions, recalls, f1s, labels):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3 * width / 8, accuracies, width/4, label='Accuracy')
    rects2 = ax.bar(x - width / 8, precisions, width/4, label='Precision')
    rects3 = ax.bar(x + width / 8, recalls, width/4, label='Recall')
    rects4 = ax.bar(x + 3 * width / 8, f1s, width/4, label='F1')

    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    fig.tight_layout()
    plt.show()


def experiment_1(plots):
    #
    #   Experiment 1: Training on subsets of collected feature data.
    #

    MFCCs = get_subsection_with_labels(feature_data, n_labels=2, from_=0, to_=26)
    YAMNet_means = get_subsection_with_labels(feature_data, n_labels=2, from_=27, to_=1051)
    YAMNet_all = get_subsection_with_labels(feature_data, n_labels=2, from_=27, to_=2075)

    feature_dict = {'MFCCs': MFCCs,
                    'YAMNet_means': YAMNet_means,
                    'YAMNet_all': YAMNet_all}

    l, a, p, r, f = [], [], [], [], []

    for key, feature in feature_dict.items():
        print('Training SVM on ' + key)
        loss_, acc_, prec_, recall_, f1_, support_ = \
            helpers.cross_validate(feature, clf, n_splits=5, n_labels=2, verbose=False)

        l.append(loss_)
        a.append(acc_)
        p.append(prec_)
        r.append(recall_)
        f.append(f1_)

    if plots:
        #   Crackle bar plot.
        accuracies = [round(a[0][0], 2), round(a[1][0], 2), round(a[2][0], 2)]
        precisions = [round(p[0][0], 2), round(p[1][0], 2), round(p[2][0], 2)]
        recalls = [round(r[0][0], 2), round(r[1][0], 2), round(r[2][0], 2)]
        f1s = [round(f[0][0], 2), round(f[1][0], 2), round(f[2][0], 2)]

        labels = ['MFCCs', 'YAMNet means', 'YAMNet all']
        bar_graph(accuracies, precisions, recalls, f1s, labels)

        #   Wheeze bar plot.
        accuracies = [round(a[0][1], 2), round(a[1][1], 2), round(a[2][1], 2)]
        precisions = [round(p[0][1], 2), round(p[1][1], 2), round(p[2][1], 2)]
        recalls = [round(r[0][1], 2), round(r[1][1], 2), round(r[2][1], 2)]
        f1s = [round(f[0][1], 2), round(f[1][1], 2), round(f[2][1], 2)]

        bar_graph(accuracies, precisions, recalls, f1s, labels)


def experiment_2(plots):
    #
    #   Experiment 2: Training on PCA reduction of total feature set.
    #

    l, a, p, r, f = [], [], [], [], []

    for i in range(10, 21, 2):
        print('\nNum PCA components = %d\n' % i)
        pca_data = helpers.get_PCA(feature_data, n_labels=2, n_components=i)
        loss_, acc_, prec_, recall_, f1_, support_ = \
            helpers.cross_validate(pca_data, clf, n_splits=5, n_labels=2, verbose=False)

        l.append(loss_)
        a.append(acc_)
        p.append(prec_)
        r.append(recall_)
        f.append(f1_)

    if plots:
        #   Crackle bar plot.
        accuracies = [round(a[0][0], 2), round(a[1][0], 2), round(a[2][0], 2),
                      round(a[3][0], 2), round(a[4][0], 2), round(a[5][0], 2)]
        precisions = [round(p[0][0], 2), round(p[1][0], 2), round(p[2][0], 2),
                      round(p[3][0], 2), round(p[4][0], 2), round(p[5][0], 2)]
        recalls = [round(r[0][0], 2), round(r[1][0], 2), round(r[2][0], 2),
                   round(r[3][0], 2), round(r[4][0], 2), round(r[5][0], 2)]
        f1s = [round(f[0][0], 2), round(f[1][0], 2), round(f[2][0], 2),
               round(f[3][0], 2), round(f[4][0], 2), round(f[5][0], 2)]

        labels = ['10', '12', '14', '16', '18', '20']
        bar_graph(accuracies, precisions, recalls, f1s, labels)

        #   Wheeze bar plot.
        accuracies = [round(a[0][1], 2), round(a[1][1], 2), round(a[2][1], 2),
                      round(a[3][1], 2), round(a[4][1], 2), round(a[5][1], 2)]
        precisions = [round(p[0][1], 2), round(p[1][1], 2), round(p[2][1], 2),
                      round(p[3][1], 2), round(p[4][1], 2), round(p[5][1], 2)]
        recalls = [round(r[0][1], 2), round(r[1][1], 2), round(r[2][1], 2),
                   round(r[3][1], 2), round(r[4][1], 2), round(r[5][1], 2)]
        f1s = [round(f[0][1], 2), round(f[1][1], 2), round(f[2][1], 2),
               round(f[3][1], 2), round(f[4][1], 2), round(f[5][1], 2)]

        bar_graph(accuracies, precisions, recalls, f1s, labels)


if __name__ == '__main__':
    clf = OneVsRestClassifier(SVC(gamma='scale'))
    helpers = RespHelpers()

    if len(sys.argv) > 1:
        if int(sys.argv[1]) == 1:
            experiment_1(plots=True)
    else:
        print('\nPlease specify which experiment to run, e.g.: \nevaluation.py 1 1 \t-->\truns both experiments.')

    if len(sys.argv) > 2:
        if int(sys.argv[2]) == 1:
            experiment_2(plots=True)

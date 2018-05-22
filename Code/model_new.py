import sys
import numpy as np
from sklearn import svm
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from visualize import plot_confusion_matrix

def distMetric(x, y, w=[]):
    #w: the weight for each dimension. w is a 1 X d array.
    d = len(x)
    distance = 0
    for i in range(d):
        distance += w[i] * (x-y)**2
    return np.sqrt(distance)

def process_line(line):
    line = line.strip().split(',')
    line = map(float, line)
    feature = line[:-1]
    label = int(line[-1])
    return (feature, label)

def pca_analysis(X, PCA=False):
    if not PCA:
        return X
    else:
        pca_components=len(X[0]) - 2
        pca_model = decomposition.PCA(n_components=pca_components)
        pca_model.fit(X)
        X = pca_model.transform(X)
        return X

def read_file(file_path):
    # read feature vectors from file
    f = open(file_path, 'r')
    line = f.readline()
    feature, label = process_line(line)
    X = [feature]
    Y = [label]
    dim_feature = len(feature)
    print("Feature vector length:" + str(dim_feature))

    for line in f:
        feature, label = process_line(line)
        X.append(feature)
        Y.append(label)
    return (X, Y)

def adaBoost_evaluate(X_train, Y_train, X_test, Y_test):
    dim_feature = len(X_train[0])
    n_estimator = 100
    adaBoost_model = AdaBoostClassifier(
            n_estimators=n_estimator)

    adaBoost_model.fit(X_train, Y_train)
    Y_predict = adaBoost_model.predict(X_test)
    acc = adaBoost_model.score(X_test, Y_test)
    cf = confusion_matrix(Y_test, Y_predict)
    recall = get_recall(Y_test, Y_predict)
    precision = get_precision(Y_test, Y_predict)
    return (adaBoost_model, acc, cf, recall, precision)

def random_forest_evaluate(X_train, Y_train, X_test, Y_test):
    dim_feature = len(X_train[0])
    n_estimator = int(dim_feature / 10)
    n_estimator = 30
    random_forest_model = RandomForestClassifier(
            n_estimators=n_estimator, #number of trees
            max_features= min(6, dim_feature - 1))
            #number of features used for each tree
    random_forest_model.fit(X_train, Y_train)
    Y_predict = random_forest_model.predict(X_test)
    acc = random_forest_model.score(X_test, Y_test)
    cf = confusion_matrix(Y_test, Y_predict)
    recall = get_recall(Y_test, Y_predict)
    precision = get_precision(Y_test, Y_predict)
    return (random_forest_model, acc, cf, recall, precision)

def knn_evaluate(X_train, Y_train, X_test, Y_test):
    knn_model = KNeighborsClassifier(
            n_neighbors=3,
            weights='uniform',
            algorithm='auto',
            metric='correlation')

    knn_model.fit(X_train,Y_train)

    Y_predict = knn_model.predict(X_test)

    acc = knn_model.score(X_test, Y_test)
    cf = confusion_matrix(Y_test, Y_predict)
    recall = get_recall(Y_test, Y_predict)
    precision = get_precision(Y_test, Y_predict)
    return (knn_model, acc, cf, recall, precision)


def bg_evaluate(X_train, Y_train, X_test, Y_test):
    bagging_model = BaggingClassifier(
            KNeighborsClassifier(),
            max_samples=0.5, max_features=0.5)
    bagging_model.fit(X_train, Y_train)
    Y_predict = bagging_model.predict(X_test)
    acc = bagging_model.score(X_test, Y_test)
    cf = confusion_matrix(Y_test, Y_predict)
    recall = get_recall(Y_test, Y_predict)
    precision = get_precision(Y_test, Y_predict)
    return (bagging_model, acc, cf, recall, precision)

def gb_evaluate(X_train, Y_train, X_test, Y_test):
    gradient_boost_model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=1.0,
            max_depth=1,
            random_state=0)

    gradient_boost_model.fit(X_train, Y_train)
    Y_predict = gradient_boost_model.predict(X_test)
    acc = gradient_boost_model.score(X_test, Y_test)
    cf = confusion_matrix(Y_test, Y_predict)
    recall = get_recall(Y_test, Y_predict)
    precision = get_precision(Y_test, Y_predict)

    return (gradient_boost_model, acc, cf, recall, precision)


def get_recall(Y_test, Y_predict):
    recall = recall_score(Y_test, Y_predict, average='macro')
    return recall

def get_precision(Y_test, Y_predict):
    precision = precision_score(Y_test, Y_predict, average='macro')
    return precision

def display_performance(acc, recall, precision):
    mean_acc = float(sum(acc))/(len(acc))
    mean_recall = float(sum(recall))/(len(recall))
    mean_precision = float(sum(precision))/(len(precision))

    acc_str = "Average Accuracy: " + '{:.3f}'.format(mean_acc)
    recall_str = " | Recall: " + '{:.3f}'.format(mean_recall)
    precision_str = " | Precision: " + '{:.3f}'.format(mean_precision)
    return acc_str + recall_str + precision_str


def main():
    classes = ['metal',
            'wood',
            'cloth',
            'plastic',
            'book']

    classes = ['glass', 'wood', 'bowl', 'coke_glass_cup']
    file_path1, file_path2 = None, None

    if len(sys.argv) == 1:
        print("Please input training & testing file names")
        print("Example:  python model.py data1.csv data2.csv")
        return

    file_name1 = sys.argv[1]
    file_path1 = '../Features/' + file_name1
    print("Training File name: " + file_name1)

    if len(sys.argv) == 3:
        file_name2 = sys.argv[2]
        print("Testing File name: " + file_name2)
        file_path2 = '../Features/' + file_name2

    # get X, Y data
    (X, Y) = read_file(file_path1)
    if len(sys.argv) == 3:
        (X2, Y2) = read_file(file_path2)

    dim_feature = len(X[0])
    n_labels = list(set(Y))
    n_label = len(set(Y))
    classes = [classes[k-1] for k in n_labels]

    #################### evaluate ##########################
    epoch = 10
    knn_acc = []
    rf_acc = []  #random forest
    gb_acc = []  #gradient boost
    bg_acc = []  #bagging model
    ad_acc = []  #adaBoost model

    knn_recall = []
    rf_recall = []  #random forest
    gb_recall = []  #gradient boost
    bg_recall = []  #bagging model
    ad_recall = []  #adaBoost model

    knn_precision = []
    rf_precision = []  #random forest
    gb_precision = []  #gradient boost
    bg_precision = []  #bagging model
    ad_precision = []  #adaBoost model

    run_knn = 1
    run_rf = 1
    run_gb = 1
    run_bg = 1
    run_ad = 1

    rf_conf_matrix = np.zeros((n_label, n_label))
    gb_conf_matrix = np.zeros((n_label, n_label))
    knn_conf_matrix = np.zeros((n_label, n_label))

    for i in range(epoch):
        # seperate train/test data for cross-validation
        X_train, X_test, Y_train, Y_test = None, None, None, None
        if len(sys.argv) == 2:
            X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y,
                    test_size=0.1,
                    random_state=None)
        else:
            X_train, Y_train = X, Y
            X_test, Y_test = X2, Y2



        #################### KNN Model ##########################
        # recreate a new model every epoch to enable cross validation
        if run_knn:
            knn_model, acc, cf, recall, precision = knn_evaluate(X_train, Y_train, X_test, Y_test)
            knn_conf_matrix += cf
            knn_acc.append(acc)
            knn_recall.append(recall)
            knn_precision.append(precision)

        #################### Random forest model ################
        # recreate a new model every epoch to enable cross validation
        if run_rf:
            random_forest_model, acc, cf, recall, precision = random_forest_evaluate(X_train, Y_train, X_test, Y_test)
            rf_conf_matrix += cf
            rf_acc.append(acc)
            rf_recall.append(recall)
            rf_precision.append(precision)

        #################### Bagging Model #####################
        if run_bg:
            bg_model, acc, cf, recall, precision = bg_evaluate(X_train, Y_train, X_test, Y_test)
            bg_acc.append(acc)
            bg_recall.append(recall)
            bg_precision.append(precision)

        #################### Gradient Boost Model ##############
        if run_gb:
            gb_model, acc, cf, recall, precision = gb_evaluate(X_train, Y_train, X_test, Y_test)
            gb_conf_matrix += cf
            gb_acc.append(acc)
            gb_recall.append(recall)
            gb_precision.append(precision)

        ################### AdaBoost Model ####################
        if run_ad:
            ad_model, acc, cf, recall, precision = adaBoost_evaluate(X_train, Y_train, X_test, Y_test)
            ad_acc.append(acc)
            ad_recall.append(recall)
            ad_precision.append(precision)

    print("- "*40)

    if run_knn:
        performance = display_performance(knn_acc, knn_recall, knn_precision)
        model = 'KNN '
        model = '{:20s}'.format(model)
        print(model + performance)

    if run_rf:
        performance = display_performance(rf_acc, rf_recall, rf_precision)
        model = 'Random Forest '
        model = '{:20s}'.format(model)
        print(model + performance)

    if run_gb:
        performance = display_performance(gb_acc, gb_recall, gb_precision)
        model = 'Gradient Boost '
        model = '{:20s}'.format(model)
        print(model + performance)

    if run_bg:
        performance = display_performance(bg_acc, bg_recall, gb_precision)
        model = 'Bagging '
        model = '{:20s}'.format(model)
        print(model + performance)

    if run_ad:
        performance = display_performance(ad_acc, ad_recall, ad_precision)
        model = 'AdaBoost '
        model = '{:20s}'.format(model)
        print(model + performance)

    print("- "*40)
    np.set_printoptions(precision=3)
    print(knn_conf_matrix)
    #plot_confusion_matrix(rf_conf_matrix, range(classes))
    #plot_confusion_matrix(gb_conf_matrix, range(classes))
    #plot_confusion_matrix(gb_conf_matrix, classes)
    #plot_confusion_matrix(knn_conf_matrix, range(classes))

    #print("Improvement of RandomForest:" + "{:.3f}".format(avg2/avg1))
    #################### evaluate ##########################

    # # model parameters for SVM
    # kernel='linear'
    # C=1
    # # create SVM model
    # svm_model = svm.SVC(kernel=kernel, C=C)
    # svm_model.fit(X_train, Y_train)
    #acc = svm_model.score(X_test, Y_test)
    #print("SVM", acc)


    #################### evaluate ##########################

    # # model parameters for SVM
    # kernel='linear'
    # C=1
    # # create SVM model
    # svm_model = svm.SVC(kernel=kernel, C=C)
    # svm_model.fit(X_train, Y_train)
    #acc = svm_model.score(X_test, Y_test)
    #print("SVM", acc)

if __name__ == "__main__":
    main()



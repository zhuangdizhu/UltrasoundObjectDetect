import sys
import numpy as np
from sklearn import svm
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
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

def main():
    file_name = 'features.csv'
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    print("File name: " + file_name)
    file_path = '../Features/' + file_name

    # read feature vectors from file
    f = open(file_path, 'r')
    line = f.readline()
    feature, label = process_line(line)
    dim_feature = len(feature)
    print("Feature vector length:" + str(dim_feature))
    X = [feature]
    Y = [label]
    dimension = len(feature)
    W = [0 for i in range(dimension)] # for KNN metric meaurement
    for line in f:
        feature, label = process_line(line)
        X.append(feature)
        Y.append(label)
        for d, f in enumerate(feature):
            W[d] += f

    for d, w in enumerate(W):
        W[d] = len(X)/w  # (1 over w/len(X))

    n_label = len(set(Y))
    ## PCA transformation.
    PCA = False
    X = pca_analysis(X, PCA)


    #################### evaluate ##########################
    epoch = 50
    knn_accuracies = []
    rf_acc = []  #random forest
    gb_acc = []  #gradient boost
    bg_acc = []  #bagging model

    run_knn = 1
    run_rf = 1
    run_gb = 0
    run_bg = 1

    conf_matrix = np.zeros((n_label, n_label))

    for i in range(epoch):
        # seperate train/test data for cross-validation
        X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y,
                test_size=0.1,
                random_state=None)

        #################### KNN Model ##########################
        # recreate a model to enable cross validation
        if run_knn:
            knn_model = KNeighborsClassifier(
                    n_neighbors=3,
                    weights='uniform',
                    algorithm='auto',
                    metric='correlation')

            knn_model.fit(X_train,Y_train)

            acc = knn_model.score(X_test, Y_test)
            knn_accuracies.append(acc)

        #################### Random forest model ################
        # recreate a model to enable cross validation
        if run_rf:
            n_estimator = int(dim_feature / 10)
            n_estimator = 30
            random_forest_model = RandomForestClassifier(
                    n_estimators=n_estimator, #number of trees
                    max_features= min(6, dim_feature - 1))
                    #max_features=int(len(X[0])/2 ))
                    #max_features=int(np.sqrt( len(X[0]))))
                    #number of features used for each tree
            random_forest_model.fit(X_train, Y_train)
            Y_predict = random_forest_model.predict(X_test)
            acc = random_forest_model.score(X_test, Y_test)
            conf_matrix += confusion_matrix(Y_test, Y_predict)
            rf_acc.append(acc)

        #################### Bagging Model #####################
        if run_bg:
            bagging_model = BaggingClassifier(
                    KNeighborsClassifier(),
                    max_samples=0.5, max_features=0.5)
            bagging_model.fit(X_train, Y_train)
            acc = bagging_model.score(X_test, Y_test)
            bg_acc.append(acc)

        #################### Gradient Boost Model ##############
        if run_gb:
            gradient_boost_model = GradientBoostingClassifier(
                    n_estimators=50,
                    learning_rate=1.0,
                    max_depth=1,
                    random_state=0)

            gradient_boost_model.fit(X_train, Y_train)
            acc = gradient_boost_model.score(X_test, Y_test)
            gb_acc.append(acc)


    print("- "*30)
    if run_knn:
        avg1 = float(sum(knn_accuracies)-min(knn_accuracies))/(len(knn_accuracies)-1)
        avg1 = float(sum(knn_accuracies))/(len(knn_accuracies))
        print("KNN Average Accuracy: " + '{:.3f}'.format(avg1))

    if run_rf:
        avg2 = float(sum(rf_acc) - min(rf_acc)) / (len(rf_acc) - 1)
        avg2 = float(sum(rf_acc)) / (len(rf_acc))
        print("Random Forest Average Accuracy: " + '{:.3f}'.format(avg2))

    if run_gb:
        avg3 = float(sum(gb_acc) - min(gb_acc)) / (len(gb_acc)-1)
        avg3 = float(sum(gb_acc)) / (len(gb_acc))
        print("Gradient Boost Average Accuracy:" + "{:.3f}".format(avg3))

    if run_bg:
        avg4 = float(sum(bg_acc) - min(bg_acc)) / (len(bg_acc)-1)
        avg4 = float(sum(bg_acc)) / (len(bg_acc))
        print("Bagging Classification Average Accuracy:" + "{:.3f}".format(avg4))

    print("- "*30)
    np.set_printoptions(precision=3)
    #print(conf_matrix)
    plot_confusion_matrix(conf_matrix, range(n_label))
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



import json
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

def load_dataset(dataset,path,obs_count=None,display=False,test_data_ratio=0.2):
    """
    Load the data from the path and create training and test data from it
    :param dataset:
    :param path:
    :param display:
    :param test_data_ratio
    :return:
    """
    dataset_path= path+"/"+dataset
    fopen = open(dataset_path)
    reviews = []
    stars = []
    count=0
    for line in fopen:
        count += 1
        a = json.loads(line)
        reviews.append(a["text"])
        stars.append(a["stars"])
        if obs_count is not None:
            if count == obs_count:
                break
    assert len(reviews) == len(stars)
    n = len(reviews)
    reviews = np.array(reviews)
    stars = np.array(stars)
    random_indices = np.arange(0, n)
    np.random.shuffle(random_indices)
    n_Train = int(n * (1-test_data_ratio))
    reviews_train = reviews[random_indices[:n_Train]]
    y_train = stars[random_indices[:n_Train]]
    reviews_test = reviews[random_indices[n_Train:]]
    y_test = stars[random_indices[n_Train:]]
    if display is True:
        print "Summary stats for dataset :" + dataset
        print "Dims Reviews Training Data: ",reviews_train.shape
        print "Dims Reviews Test Data: ", reviews_train.shape
        print "Sample features data: ", reviews_train[0:2]
        print "Distinct Classes: ", np.unique(y_train)
    return reviews_train, y_train, reviews_test,y_test


def get_tfidf_features(reviews_train,reviews_test):
    """
    :param reviews_train:
    :param reviews_test:
    :return:
    """
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(reviews_train)
    tfidf_transformer = TfidfTransformer()
    x_train = tfidf_transformer.fit_transform(X_train_counts)
    print "Dims of Word Count Vectorizer of Training Data: ",X_train_counts.shape
    print "Dims of TF-IDF Vectorizer of Training Data: ", x_train.shape
    X_Test_counts = count_vect.transform(reviews_test)
    x_test = tfidf_transformer.transform(X_Test_counts)
    print "Dims of Word Count Vectorizer of Test Data: ", X_Test_counts.shape
    print "Dims of TF-IDF Vectorizer of Test Data: ", x_test.shape
    return x_train,x_test


def get_raw_classifier_accuracy(classifier_type,x_train,y_train):
    """
    :param classifier_type:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    if classifier_type == 'logit':
        print "logit"
        clf = LogisticRegression(n_jobs=4)
        clf.fit(x_train, y_train)
        cv_scores= (cross_validation.cross_val_score(clf, x_train,y_train, cv=5)).mean()
    elif classifier_type == 'svm':
        print "svm"
        clf = svm.SVC(kernel='linear',)
        clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'mnb':
        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'bnb':
        clf = BernoulliNB()
        clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'gnb':
        clf = GaussianNB()
        clf = clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'rf':
        print "Rf"
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy', oob_score=True,n_jobs=4)
        clf = clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'gbt':
        clf = GradientBoostingClassifier(verbose=True)
        clf = clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'lda':
        clf = LinearDiscriminantAnalysis()
        clf = clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'qda':
        clf = QuadraticDiscriminantAnalysis()
        clf = clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'dt':
        clf = DecisionTreeClassifier(max_depth=5)
        clf = clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'knn':
        print "knn"
        clf = KNeighborsClassifier(n_neighbors=10,n_jobs=4)
        clf = clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    elif classifier_type == 'ada':
        clf = AdaBoostClassifier()
        clf = clf.fit(x_train, y_train)
        cv_scores = (cross_validation.cross_val_score(clf, x_train, y_train, cv=5)).mean()
    print "Cross-validation accuracy for the data using classifier: %s  : %f" %(classifier_type,cv_scores)
    return cv_scores


def get_evaluation_metrics(y_test,y_predicted):
    """
    :param y_test:
    :param y_predicted:
    :return:
    """
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    print "classes: ",lb.classes_, "\n\n\n"
    Y_test_b = lb.transform(y_test)
    predicted_b = lb.transform(y_predicted)
    accuracy = np.mean(y_test == y_predicted)
    print "Accuracy: ", accuracy
    print "Precision: ", average_precision_score (Y_test_b, predicted_b, average='micro')
    print "Recall: ", recall_score(Y_test_b, predicted_b, average='micro')
    print "f1-score: ", f1_score(Y_test_b, predicted_b, average='micro')
    return accuracy


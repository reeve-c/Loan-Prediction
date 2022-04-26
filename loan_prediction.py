import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report

def prepare_dataset(dataset):

    # Dropping Columns
    dataset.drop('Loan_ID', axis=1, inplace=True)

    # Taking Care of Missing Values
    null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount', 'Dependents',
                 'Loan_Amount_Term', 'Gender', 'Married']
    for col in null_cols:
        # Replacing NaN values with Mode of the column
        dataset[col] = dataset[col].fillna(dataset[col].dropna().mode().values[0])


    # Converting String Variables to Integer
    dataset['Dependents'].replace(to_replace='3+', value='3', inplace=True)
    dataset['Dependents'] = dataset['Dependents'].astype(int)

    # Encoding Categorical Variables
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # Columns to be encoded
    to_numeric = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

    for col in to_numeric:
        dataset[col] = le.fit_transform(dataset[col])

    # Splitting the Dataset into Train and Test Sets
    X = dataset.drop('Loan_Status', axis=1)
    Y = dataset['Loan_Status']

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    return X_train, X_test, Y_train, Y_test

def logistic_regression(X_train, X_test, Y_train, Y_test):
    from sklearn.linear_model import LogisticRegression

    # Initializing the Model
    regerssor = LogisticRegression()

    # Fitting the Training Data
    regerssor.fit(X_train, Y_train)

    # Predicting on Test Set
    Y_pred = regerssor.predict(X_test)

    # Evaluating the Model
    lr_ac = accuracy_score(Y_pred, Y_test)
    print('\n'+'-'*15+" LOGISTIC REGRESSION "+'-'*50+'\n')
    print(f'Accuracy : {lr_ac}\n')
    print(f"Classification Report: \n{classification_report(Y_test, Y_pred)}")

    return lr_ac

def decision_tree(X_train, X_test, Y_train, Y_test):
    from sklearn.tree import DecisionTreeClassifier

    # Initializing the Model
    dt = DecisionTreeClassifier()

    # Fitting the Training Data
    dt.fit(X_train, Y_train)

    # Predicting on Test Set
    Y_pred = dt.predict(X_test)

    # Evaluating the Model
    dt_ac = accuracy_score(Y_pred, Y_test)
    print('-'*15+" DECISION TREE CLASSIFIER "+'-'*50+'\n')
    print(f'Accuracy : {dt_ac}\n')
    print(f"Classification Report: \n{classification_report(Y_test, Y_pred)}")

    return dt_ac

def random_forest(X_train, X_test, Y_train, Y_test):
    from sklearn.ensemble import RandomForestClassifier

    # Initializing the Model
    rfc = RandomForestClassifier()

    # Fitting the Training Data
    rfc.fit(X_train, Y_train)

    # Predicting on Test Set
    Y_pred = rfc.predict(X_test)

    # Evaluating the Model
    rfc_ac = accuracy_score(Y_pred, Y_test)
    print('-'*15+" RANDOM FOREST CLASSIFIER "+'-'*50+'\n')
    print(f'Accuracy : {rfc_ac}\n')
    print(f"Classification Report: \n{classification_report(Y_test, Y_pred)}")

    return rfc_ac

def svm(X_train, X_test, Y_train, Y_test):
    from sklearn.svm import SVC

    # Initializing the Model
    svm = SVC(kernel='linear', C=0.025)

    # Fitting the Training Data
    svm.fit(X_train, Y_train)

    # Predicting on Test Set
    Y_pred = svm.predict(X_test)

    # Evaluating the Model
    svm_ac = accuracy_score(Y_pred, Y_test)
    print('-'*15+" SUPPORT VECTOR MACHINE "+'-'*50+'\n')
    print(f'Accuracy : {svm_ac}\n')
    print(f"Classification Report: \n{classification_report(Y_test, Y_pred)}")

    return svm_ac

def naive_bayes(X_train, X_test, Y_train, Y_test):
    from sklearn.naive_bayes import GaussianNB

    # Initializing the Model
    nb = GaussianNB()

    # Fitting the Training Data
    nb.fit(X_train, Y_train)

    # Predicting on Test Set
    Y_pred = nb.predict(X_test)

    # Evaluating the Model
    nb_ac = accuracy_score(Y_pred, Y_test)
    print('-'*15+" NAIVE BAYES "+'-'*50+'\n')
    print(f'Accuracy : {nb_ac}\n')
    print(f"Classification Report: \n{classification_report(Y_test, Y_pred)}")

    return nb_ac

def knn(X_train, X_test, Y_train, Y_test):
    from sklearn.neighbors import KNeighborsClassifier

    # Initializing the Model
    knn = KNeighborsClassifier(n_neighbors=30)

    # Fitting the Training Data
    knn.fit(X_train, Y_train)

    # Predicting on Test Set
    Y_pred = knn.predict(X_test)

    # Evaluating the Model
    knn_ac = accuracy_score(Y_pred, Y_test)
    print('-'*15+" K-NEAREST NEIGHBORS "+'-'*50+'\n')
    print(f'Accuracy : {knn_ac}\n')
    print(f"Classification Report: \n{classification_report(Y_test, Y_pred)}")

    return knn_ac

if __name__ == "__main__":

    #  Importing the Dataset
    dataset = pd.read_csv('loan_prediction_dataset.csv')


    # Preparing the Data
    X_train, X_test, Y_train, Y_test = prepare_dataset(dataset)

    # Predicting Using Various Models
    lr_ac = logistic_regression(X_train, X_test, Y_train, Y_test)
    dt_ac = decision_tree(X_train, X_test, Y_train, Y_test)
    rfc_ac = random_forest(X_train, X_test, Y_train, Y_test)
    svm_ac = svm(X_train, X_test, Y_train, Y_test)
    nb_ac = naive_bayes(X_train, X_test, Y_train, Y_test)
    knn_ac = knn(X_train, X_test, Y_train, Y_test)

    print('-'*10+" Comparing Accuracy of all the Models: "+'-'*10+'\n')
    print(pd.DataFrame(zip(['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'Naive Bayes',
                      'K-Nearest Neighbor'],
                     [round(lr_ac, 2), round(dt_ac, 2), round(rfc_ac, 2), round(svm_ac, 2), round(nb_ac, 2),
                      round(knn_ac, 2)]), columns=['Model', 'Accuracy']))

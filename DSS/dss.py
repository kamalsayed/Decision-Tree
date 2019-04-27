from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np


X,y = load_digits(n_class=10,return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


def knn_cross_val(Xtr,ytr,Xt,yt):
    knn_cv = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 25)}
    knn_gscv = GridSearchCV(knn_cv, param_grid, cv=5)
    knn_gscv.fit(Xtr, ytr)
    print("the best k is : ",knn_gscv.best_params_)
    print("accuracy for knn with best k : ",knn_gscv.score(Xt,yt))


def logistic_reg(Xtr,ytr,Xt,yt):
    LR=LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / Xtr.shape[0],multi_class='auto').fit(Xtr,ytr)  
    print("Logisitc regression accuracy : ",LR.score(Xt,yt))

def decision_tree(Xtr,ytr,Xt,yt):
    dt=DecisionTreeClassifier()
    dt.fit(Xtr,ytr)
    print("decition tree accuracy is : ",dt.score(Xt,yt))

knn_cross_val(X_train,y_train,X_test,y_test)
logistic_reg(X_train,y_train,X_test,y_test)
decision_tree(X_train,y_train,X_test,y_test)
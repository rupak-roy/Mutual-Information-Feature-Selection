#mutual Information Classification

# load cancer data
from sklearn.datasets import load_breast_cancer as LBC
cancer = LBC()
X = cancer['data']
y = cancer['target']

#Compute MI scores
from sklearn.feature_selection import mutual_info_classif as MIC
mi_scores = MIC(X,y)
print(mi_scores)

#prepare dataset 1
from sklearn.model_selection import train_test_split as tts
X_train_1,X_test_1,y_train,y_test = tts(
    X,y,random_state=0,stratify=y )

# prepare dataset 2, MI > 0.2
import numpy as np
mi_score_selected_index = np.where(mi_scores >0.2)[0]
X_2 = X[:,mi_score_selected_index]
X_train_2,X_test_2,y_train,y_test = tts(
    X_2,y,random_state=0,stratify=y)

# prepare dataset 3, MI <0.2
mi_score_selected_index = np.where(mi_scores < 0.2)[0]
X_3 = X[:,mi_score_selected_index]
X_train_3,X_test_3,y_train,y_test = tts(
    X_3,y,random_state=0,stratify=y)

# compare results with Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier as DTC
model_1 = DTC().fit(X_train_1,y_train)
model_2 = DTC().fit(X_train_2,y_train)
model_3 = DTC().fit(X_train_3,y_train)
score_1 = model_1.score(X_test_1,y_test)
score_2 = model_2.score(X_test_2,y_test)
score_3 = model_3.score(X_test_3,y_test)

# use Scikit-learn feature selector
from sklearn.feature_selection import SelectPercentile as SP
selector = SP(percentile=50) # select features with top 50% MI scores
selector.fit(X,y)
X_4 = selector.transform(X)
X_train_4,X_test_4,y_train,y_test = tts(
    X_4,y,random_state=0,stratify=y)

model_4 = DTC().fit(X_train_4,y_train)
score_4 = model_4.score(X_test_4,y_test)
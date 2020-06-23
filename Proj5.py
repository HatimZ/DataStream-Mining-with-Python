# programmer : Hatim Zahid 21603260
# 5/24/2020
#

#Q1
import numpy as np
from skmultiflow.data import RandomRBFGeneratorDrift
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
import pandas as pd
from sklearn.neural_network import MLPClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import FileStream, RandomRBFGenerator, RandomRBFGeneratorDrift
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.meta import BatchIncremental
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data import SEAGenerator
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import FileStream, RandomRBFGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.meta import BatchIncremental
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data import SEAGenerator
from sklearn.ensemble import VotingClassifier


stream2 = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state = 50, n_classes = 2, n_features = 10,
n_centroids = 10000, change_speed= 10)
X, y = stream2.next_sample(10000)
stream2.restart()
df2 = pd.DataFrame(np.hstack((X,np.array([y]).T)))
df2.to_csv("RBF Dataset 10.csv")

stream3 = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state = 50, n_classes = 2, n_features = 10,
n_centroids = 10000, change_speed= 70)
X, y = stream3.next_sample(10000)
stream3.restart()
df3 = pd.DataFrame(np.hstack((X,np.array([y]).T)))
df3.to_csv("RBF Dataset 70.csv")

#Single Online Classifiers
stream10 = FileStream("./"+'RBF Dataset 10'+'.csv')
stream70 = FileStream("./"+'RBF Dataset 70'+'.csv')
stream = FileStream("./"+'RBF Dataset'+'.csv')

MLP = MLPClassifier(hidden_layer_sizes=(200,200,200,200 ) ,random_state=1, max_iter=500)
nb = NaiveBayes()
ht = HoeffdingTreeClassifier()

evaluator = EvaluatePrequential( max_samples=10000,
max_time=1000,
show_plot=True,
pretrain_size= 3000,
metrics=['accuracy'])

#Ensemble Online
stream10 = FileStream("./"+'RBF Dataset 10'+'.csv')
stream70 = FileStream("./"+'RBF Dataset 70'+'.csv')
stream = FileStream("./"+'RBF Dataset'+'.csv')

MLP = MLPClassifier(hidden_layer_sizes=(200,200,200,200 ) ,random_state=1, max_iter=500)
nb = NaiveBayes()
ht = HoeffdingTreeClassifier()

evaluator = EvaluatePrequential( max_samples=10000,
max_time=1000,
show_plot=True,
pretrain_size= 3000,
metrics=['accuracy'])

#Single Bathc Classification
#stream = FileStream('file.csv')
stream = FileStream("RBF Dataset.csv")
stream10 = FileStream("RBF Dataset 10.csv")
stream70 = FileStream("RBF Dataset 70.csv")

X1,y1 = stream.next_sample(10000)
X2,y2 = stream10.next_sample(10000)
X3,y3 = stream70.next_sample(10000)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size= 0.4,random_state=109)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size= 0.4,random_state=109)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size= 0.4,random_state=109)



#Classifiers
MLP = MLPClassifier(hidden_layer_sizes=(200, 4 ) ,random_state=1, max_iter=300)
nb = NaiveBayes()
ht = HoeffdingTreeClassifier()

# ht2_trained = ht.fit(X3_train,y3_train)
# ht2_predictions =  ht.predict(X3_test)
# print("Below are the evaluations on Validating Set /n")
# print(confusion_matrix(y1_test , ht1_predictions))
# print(classification_report(y3_test , ht2_predictions))

# nb_trained = nb.fit(X3_train,y3_train)
# nb_predictions =  nb.predict(X3_test)
# print("Below are the evaluations on Validating Set /n")
# # print(confusion_matrix(y1_test , ht1_predictions))
# print(classification_report(y3_test , nb_predictions))

MLP_trained = MLP.fit(X3_train,y3_train)
MLP_predictions =  MLP.predict(X3_test)
print("Below are the evaluations on Validating Set /n")
# print(confusion_matrix(y1_test , ht1_predictions))
print(classification_report(y3_test , MLP_predictions))


#Ensemble Batch

stream = FileStream("RBF Dataset.csv")
stream10 = FileStream("RBF Dataset 10.csv")
stream70 = FileStream("RBF Dataset 70.csv")

X1,y1 = stream.next_sample(10000)
X2,y2 = stream10.next_sample(10000)
X3,y3 = stream70.next_sample(10000)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size= 0.4,random_state=109)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size= 0.4,random_state=109)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size= 0.4,random_state=109)



MLP = MLPClassifier(hidden_layer_sizes=(200, 4) ,random_state=1, max_iter=1200)
ht = HoeffdingTreeClassifier()
nb = NaiveBayes()

#DWM = DynamicWeightedMajorityClassifier(n_estimators=5, base_estimator= nb, period=50, beta=0.5, theta=0.01)
estimators = []
estimators.append(('nb', nb))
estimators.append(('Ht', ht))
estimators.append(('MLP', MLP))
# create the ensemble model
ensemble = VotingClassifier(estimators , voting= 'hard')
ensemble_wm = VotingClassifier(estimators , voting= 'soft')

# ensemble_trained = ensemble.fit(X2_train,y2_train)
# ensemble_predictions =  ensemble.predict(X2_test)
# print("Below are the evaluations on Validating Set /n")
# # print(confusion_matrix(y1_test , ht1_predictions))
# print(classification_report(y2_test , ensemble_predictions))

ensemble_wm_trained = ensemble_wm.fit(X3_train,y3_train)
ensemble_wm_predictions =  ensemble_wm.predict(X3_test)
print("Below are the evaluations on Validating Set /n")
# print(confusion_matrix(y1_test , ht1_predictions))
print(classification_report(y3_test , ensemble_wm_predictions))
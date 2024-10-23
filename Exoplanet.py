import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import  LabelEncoder , Normalizer 
from sklearn.model_selection import  KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, classification_report, 
                             precision_score, recall_score, matthews_corrcoef,
                             cohen_kappa_score, log_loss, precision_recall_curve,
                             RocCurveDisplay, PrecisionRecallDisplay , roc_curve)


import torch as T
import torch.nn as nn
import torch.optim as O
from torch.utils.data import DataLoader, TensorDataset

import tqdm 
import os

df = pd.read_csv("../csvs/Space/dataset.csv")
df
df = df.drop(["prefix" , "name", "albedo", "diameter_sigma", "diameter"] , axis = 1)


df["sigma_ad"].dtype

si = SimpleImputer(strategy= "mean")
sio = SimpleImputer(strategy= "most_frequent")

df[["sigma_ad"]] = si.fit_transform(df[["sigma_ad"]])
df[["sigma_per"]] = si.fit_transform(df[["sigma_per"]])
df[["sigma_e"]] = si.fit_transform(df[["sigma_e"]])
df[["sigma_a"]] = si.fit_transform(df[["sigma_a"]])
df[["sigma_q"]] = si.fit_transform(df[["sigma_q"]])
df[["sigma_i"]] = si.fit_transform(df[["sigma_i"]])
df[["sigma_om"]] = si.fit_transform(df[["sigma_om"]])
df[["sigma_w"]] = si.fit_transform(df[["sigma_w"]])
df[["sigma_ma"]] = si.fit_transform(df[["sigma_ma"]])
df[["sigma_n"]] = si.fit_transform(df[["sigma_n"]])
df[["sigma_tp"]] = si.fit_transform(df[["sigma_tp"]])
df[["pha"]] = sio.fit_transform(df[["pha"]])
df[["moid"]] = si.fit_transform(df[["moid"]])
df[["H"]] = si.fit_transform(df[["H"]])
df[["moid_ld"]] = si.fit_transform(df[["moid_ld"]])
df[["per"]] = si.fit_transform(df[["per"]])
df[["ad"]] = si.fit_transform(df[["ad"]])
df[["neo"]] = sio.fit_transform(df[["neo"]])
df[["rms"]] = si.fit_transform(df[["rms"]])
df[["ma"]] = si.fit_transform(df[["ma"]])
df[["per_y"]] = si.fit_transform(df[["per_y"]])


df.isnull().sum().sort_values(ascending=False)

pd.set_option('display.max_columns', None)

le = LabelEncoder()
df["pha"] = le.fit_transform(df["pha"])
df["neo"] = le.fit_transform(df["neo"])
df["orbit_id"] = le.fit_transform(df["orbit_id"])
df["full_name"] = le.fit_transform(df["full_name"])
df["id"] = le.fit_transform(df["id"])
df["equinox"] = le.fit_transform(df["equinox"])
df["class"] = le.fit_transform(df["class"])

df["pdes"] = df["pdes"].astype(str)
df["pdes"] = le.fit_transform(df["pdes"])

X = df.drop(["pha"], axis = 1).values
print(X.shape)
X = X.astype(np.float32)
X

y = df["pha"].values.reshape(-1,1)
print(y.shape)
y = y.astype(np.float32)
y

fold = KFold(n_splits=10 , shuffle=True)
for train , test in fold.split(X , y):
    X_train , X_test = X[train] , X[test]
    y_train , y_test = y[train], y[test]

# Vor Skalierung
print("Vor der Skalierung:")
print(f"Trainingsdaten - Min: {X_train.min()}, Max: {X_train.max()}")
print(f"Testdaten - Min: {X_test.min()}, Max: {X_test.max()}")

sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Vor Skalierung
print("Vor der Skalierung:")
print(f"Trainingsdaten - Min: {X_train.min()}, Max: {X_train.max()}")
print(f"Testdaten - Min: {X_test.min()}, Max: {X_test.max()}")

unique , counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique , counts)))

smote = SMOTE()
X_train , y_train = smote.fit_resample(X_train , y_train)

unique , counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique , counts)))

print(type(X_train))
print(type(X_test))
print(type(y_train))
print(type(y_test))

X_train = T.from_numpy(X_train)
X_test = T.from_numpy(X_test)
y_train = T.from_numpy(y_train)
y_test = T.from_numpy(y_test)

print(type(X_train))
print(type(X_test))
print(type(y_train))
print(type(y_test))

class NN(nn.Module):

    def __init__(self, in_dim = X_train.shape[1] , out_dim = 1):
        super().__init__()
        self.ll1 = nn.Linear(in_dim , 20)
        self.ll2 = nn.Linear(20 , 25 )

        self.ll3 = nn.Linear(25 , 25)
        self.ll4 = nn.Linear(25 , out_dim)

        self.drop = nn.Dropout(p = (0.3))
        self.activation = nn.PReLU()

    def forward(self , X):

        X = self.activation(self.ll1(X))
        X = self.activation(self.ll2(X))
        X = self.drop(X)
        X = self.activation(self.ll3(X))
        X = self.drop(X)
        X = self.ll4(X)

        return X

class training():

    def __init__(self):
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.model = NN().to(self.device)
        self.epochs = 20
        self.lr = 1e-4
        self.momentum = 0.9
        self.crit = O.SGD(self.model.parameters() , lr = self.lr, weight_decay= 1e-3)
        self.crit2 = O.Adam(self.model.parameters() , self.lr , weight_decay= 1e-4)
        self.loss = nn.BCEWithLogitsLoss()

        self.X_train = X_train.float()
        self.X_test = X_test.float()
        self.y_train = y_train.float()
        self.y_test = y_test.float()

        self.batch_size = 64

        self.train_loader = DataLoader(dataset= TensorDataset(self.X_train , self.y_train),
                                       shuffle=True,
                                       batch_size=self.batch_size ,
                                       num_workers= os.cpu_count())
        
        self.test_loader = DataLoader(dataset= TensorDataset(self.X_test , self.y_test),
                               batch_size=self.batch_size ,
                               num_workers= os.cpu_count())
        
    def train_loop(self):

        for i in range(self.epochs):
            self.model.train()
            current_loss = 0.0
            correct_train = 0.0
            with tqdm.tqdm(iterable=self.train_loader , mininterval=0.1 , disable=False) as pbar:
                for X , y in pbar:
                    X ,y = X.to(self.device) , y.to(self.device)
                    pbar.set_description(f"epoch : {i + 1}")
                    logits = self.model(X)
                    loss = self.loss(logits , y.reshape(-1,1))
                    self.crit2.zero_grad()
                    loss.backward()
                    self.crit2.step()

                    current_loss += loss.item()
                    pbar.set_postfix({"loss on current batch" : loss.item()})

                print("epoch [{}/{}], iteration [{}/{}], TrainLoss = {:.4f}, train_acc: {:.2f}%".format(
                    i + 1, 
                    i, 
                    i + 1, 
                    len(self.train_loader) // self.batch_size, 
                    loss.data[0], 
                    100 * correct_train.double()/len(self.train_loader) 
                ))


    def evaluate_model(self):
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probs = []
    
        with T.no_grad(): 
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                probabilities = T.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()  

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
    
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        all_probs = np.array(all_probs).flatten()  
    
        accuracy = accuracy_score(all_labels, all_predictions)
        confusion = confusion_matrix(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        roc_auc = roc_auc_score(all_labels, all_probs) 
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted')
        mcc = matthews_corrcoef(all_labels, all_predictions)
        cohen_kappa = cohen_kappa_score(all_labels, all_predictions)
        log_loss_value = log_loss(all_labels, all_probs)
    
        print("Evaluationsmetriken:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Cohen Kappa: {cohen_kappa:.4f}")
        print(f"Log Loss: {log_loss_value:.4f}")
        print("Confusion Matrix:")
        print(confusion)
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))
    
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

        PrecisionRecallDisplay.from_predictions(all_labels, all_probs)  
        plt.show()

training().train_loop()
training().evaluate_model()

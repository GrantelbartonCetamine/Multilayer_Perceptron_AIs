class NN(nn.Module):

    def __init__(self, in_dim = X_train.shape[1] , out_dim = 1):
        super().__init__()
        self.ll1 = nn.Linear(in_dim , 5)
        self.ll2 = nn.Linear(5 , 5 )

        self.ll3 = nn.Linear(5 , 8)
        self.ll4 = nn.Linear(8 , out_dim)

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
        self.epochs = 10
        self.lr = 1e-3
        self.momentum = 0.9
        self.crit = O.SGD(self.model.parameters() , lr = self.lr, weight_decay= 1e-3 , momentum = self.momemtum)
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
        
        self.test_loader = DataLoader(dataset = TensorDataset(self.X_test , self.y_test),
                               batch_size=self.batch_size ,
                               num_workers= os.cpu_count())
        
    def train_loop(self):

        for i in range(self.epochs):
            self.model.train()
            current_loss = 0.0
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

            print(f"Epoch {i + 1}/{self.epochs} - Loss Overall: {current_loss / len(self.train_loader):.4f}") 

    def evaluate_model(self):
        self.model.eval()  # Setze das Modell in den Evaluierungsmodus
    
        # Variablen zur Speicherung der Vorhersagen und wahren Labels
        all_predictions = []
        all_labels = []
        all_probs = []
    
        with T.no_grad():  # Keine Gradientenberechnung nötig
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                # Vorhersagen des Modells
                outputs = self.model(inputs)
                probabilities = T.sigmoid(outputs)  # Für Wahrscheinlichkeiten
                predictions = (probabilities > 0.5).float()  # binäre Vorhersagen
    
                # Speichere die Vorhersagen und Labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
    
        # Konvertiere Listen in Numpy Arrays
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        all_probs = np.array(all_probs).flatten()  # Flatten, um sicherzustellen, dass es eine Dimension hat
    
        # Berechne die Metriken
        accuracy = accuracy_score(all_labels, all_predictions)
        confusion = confusion_matrix(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        roc_auc = roc_auc_score(all_labels, all_probs)  # AUC für binäre Klassifikation
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted')
        mcc = matthews_corrcoef(all_labels, all_predictions)
        cohen_kappa = cohen_kappa_score(all_labels, all_predictions)
        log_loss_value = log_loss(all_labels, all_probs)
    
        # Ausgabe der Metriken
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
    
        # ROC-Kurve und Precision-Recall-Diagramm
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        
        # Sicherstellen, dass all_probs die Wahrscheinlichkeiten für die positive Klasse sind
        PrecisionRecallDisplay.from_predictions(all_labels, all_probs)  # Verwende hier all_probs
    
        plt.show()

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
import time

import pandas as pd

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)
torch.manual_seed(7)

#load the data into a dataframe
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# get rid of columns like id and names that have no impact
train_df = train_df[["Patient Age", 
                    "Genes in mother's side", 
                    "Inherited from father", 
                    "Maternal gene", 
                    "Paternal gene", 
                    "Blood cell count (mcL)",
                    "Status", 
                    "Respiratory Rate (breaths/min)", 
                    "Heart Rate (rates/min", 
                    "Test 1",
                    "Test 2",
                    "Test 3",
                    "Test 4",
                    "Test 5",
                    "Follow-up", 
                    "Gender", 
                    "Birth asphyxia", 
                    "Autopsy shows birth defect (if applicable)",
                    "Folic acid details (peri-conceptional)", 
                    "H/O serious maternal illness", 
                    "H/O radiation exposure (x-ray)", 
                    "H/O substance abuse", 
                    "Assisted conception IVF/ART", 
                    "History of anomalies in previous pregnancies", 
                    "No. of previous abortion", 
                    "Birth defects", 
                    "White Blood cell count (thousand per microliter)", 
                    "Blood test result", 
                    "Symptom 1", 
                    "Symptom 2", 
                    "Symptom 3", 
                    "Symptom 4", 
                    "Symptom 5", 
                    "Genetic Disorder", 
                    "Disorder Subclass"]]

# Remove rows missing both targets
train_df = train_df[(train_df["Genetic Disorder"].isnull() != True) & (train_df["Disorder Subclass"].isnull() != True)]

# replace no responses with nan
train_df = train_df.replace('-', np.nan)

# replace no responses or invalid inputs with nan
test_df = test_df.replace('-', np.nan)
test_df = test_df.replace('-99', np.nan)

# Final columns
train_df = train_df[["Patient Age", 
                    "Genes in mother's side", 
                    "Inherited from father", 
                    "Maternal gene", 
                    "Paternal gene", 
                    "Blood cell count (mcL)",
                    "Status", 
                    "Respiratory Rate (breaths/min)", 
                    "Heart Rate (rates/min", 
                    "Follow-up", 
                    "Gender", 
                    "Folic acid details (peri-conceptional)", 
                    "H/O serious maternal illness", 
                    "Assisted conception IVF/ART", 
                    "History of anomalies in previous pregnancies", 
                    "No. of previous abortion", 
                    "Birth defects", 
                    "White Blood cell count (thousand per microliter)", 
                    "Blood test result", 
                    "Symptom 1", 
                    "Symptom 2", 
                    "Symptom 3", 
                    "Symptom 4", 
                    "Symptom 5", 
                    "Genetic Disorder", 
                    "Disorder Subclass"]]

test_df = test_df[["Patient Age", 
                    "Genes in mother's side", 
                    "Inherited from father", 
                    "Maternal gene", 
                    "Paternal gene", 
                    "Blood cell count (mcL)",
                    "Status", 
                    "Respiratory Rate (breaths/min)", 
                    "Heart Rate (rates/min", 
                    "Follow-up", 
                    "Gender", 
                    "Folic acid details (peri-conceptional)", 
                    "H/O serious maternal illness", 
                    "Assisted conception IVF/ART", 
                    "History of anomalies in previous pregnancies", 
                    "No. of previous abortion", 
                    "Birth defects", 
                    "White Blood cell count (thousand per microliter)", 
                    "Blood test result", 
                    "Symptom 1", 
                    "Symptom 2", 
                    "Symptom 3", 
                    "Symptom 4", 
                    "Symptom 5"]]

# delete rows with 5 or more missing variables (in each row)
train_df = train_df[train_df.isnull().sum(axis=1) < 5] 

# replace nans with the most frequent responses in that column
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

for col in train_df.columns:
    imp.fit(train_df[[col]])
    train_df[[col]] = imp.transform(train_df[[col]])

#training data
#switch yes/no responses to true/false to create booleans
train_df["Genes in mother's side"] = train_df["Genes in mother's side"].map(dict(Yes=True, No=False))
train_df["Inherited from father"] = train_df["Inherited from father"].map(dict(Yes=True, No=False))
train_df["Maternal gene"] = train_df["Maternal gene"].map(dict(Yes=True, No=False))
train_df["Paternal gene"] = train_df["Paternal gene"].map(dict(Yes=True, No=False))
train_df["Folic acid details (peri-conceptional)"] = train_df["Folic acid details (peri-conceptional)"].map(dict(Yes=True, No=False))
train_df["H/O serious maternal illness"] = train_df["H/O serious maternal illness"].map(dict(Yes=True, No=False))
train_df["Assisted conception IVF/ART"] = train_df["Assisted conception IVF/ART"].map(dict(Yes=True, No=False))
train_df["History of anomalies in previous pregnancies"] = train_df["History of anomalies in previous pregnancies"].map(dict(Yes=True, No=False))


#test data
#switch yes/no responses to true/false to create booleans
test_df["Genes in mother's side"] = test_df["Genes in mother's side"].map(dict(Yes=True, No=False))
test_df["Inherited from father"] = test_df["Inherited from father"].map(dict(Yes=True, No=False))
test_df["Maternal gene"] = test_df["Maternal gene"].map(dict(Yes=True, No=False))
test_df["Paternal gene"] = test_df["Paternal gene"].map(dict(Yes=True, No=False))
test_df["Folic acid details (peri-conceptional)"] = test_df["Folic acid details (peri-conceptional)"].map(dict(Yes=True, No=False))
test_df["H/O serious maternal illness"] = test_df["H/O serious maternal illness"].map(dict(Yes=True, No=False))
test_df["Assisted conception IVF/ART"] = test_df["Assisted conception IVF/ART"].map(dict(Yes=True, No=False))
test_df["History of anomalies in previous pregnancies"] = test_df["History of anomalies in previous pregnancies"].map(dict(Yes=True, No=False))

#training data
#set each column to its appropriate datatype
train_df["Patient Age"]  = train_df["Patient Age"].astype('int64')
train_df["Genes in mother's side"]  = train_df["Genes in mother's side"].astype('bool')
train_df["Inherited from father"]  = train_df["Inherited from father"].astype('bool')
train_df["Maternal gene"]  = train_df["Maternal gene"].astype('bool')
train_df["Paternal gene"]  = train_df["Paternal gene"].astype('bool')
train_df["Folic acid details (peri-conceptional)"]  = train_df["Folic acid details (peri-conceptional)"].astype('bool')
train_df["H/O serious maternal illness"]  = train_df["H/O serious maternal illness"].astype('bool')
train_df["Assisted conception IVF/ART"]  = train_df["Assisted conception IVF/ART"].astype('bool')
train_df["History of anomalies in previous pregnancies"]  = train_df["History of anomalies in previous pregnancies"].astype('bool')
train_df["No. of previous abortion"]  = train_df["No. of previous abortion"].astype('int64')
train_df["Symptom 1"]  = train_df["Symptom 1"].astype('bool')
train_df["Symptom 2"]  = train_df["Symptom 2"].astype('bool')
train_df["Symptom 3"]  = train_df["Symptom 3"].astype('bool')
train_df["Symptom 4"]  = train_df["Symptom 4"].astype('bool')
train_df["Symptom 5"]  = train_df["Symptom 5"].astype('bool')

#create dummy values for categorical data
train_df = pd.get_dummies(train_df, columns=["Status", 
                    "Respiratory Rate (breaths/min)", 
                    "Heart Rate (rates/min", 
                    "Follow-up", 
                    "Gender", 
                    "Birth defects",
                    "Blood test result"])



#testing data
#set each column to its appropriate datatype
test_df["Genes in mother's side"]  = test_df["Genes in mother's side"].astype('boolean')
test_df["Inherited from father"]  = test_df["Inherited from father"].astype('boolean')
test_df["Maternal gene"]  = test_df["Maternal gene"].astype('boolean')
test_df["Paternal gene"]  = test_df["Paternal gene"].astype('boolean')
test_df["Folic acid details (peri-conceptional)"]  = test_df["Folic acid details (peri-conceptional)"].astype('boolean')
test_df["H/O serious maternal illness"]  = test_df["H/O serious maternal illness"].astype('boolean')
test_df["Assisted conception IVF/ART"]  = test_df["Assisted conception IVF/ART"].astype('boolean')
test_df["History of anomalies in previous pregnancies"]  = test_df["History of anomalies in previous pregnancies"].astype('boolean')
test_df["Symptom 1"]  = test_df["Symptom 1"].astype('boolean')
test_df["Symptom 2"]  = test_df["Symptom 2"].astype('boolean')
test_df["Symptom 3"]  = test_df["Symptom 3"].astype('boolean')
test_df["Symptom 4"]  = test_df["Symptom 4"].astype('boolean')
test_df["Symptom 5"]  = test_df["Symptom 5"].astype('boolean')

#create dummy values for categorical data
test_df = pd.get_dummies(test_df, columns=["Status", 
                    "Respiratory Rate (breaths/min)", 
                    "Heart Rate (rates/min", 
                    "Follow-up", 
                    "Gender", 
                    "Birth defects",
                    "Blood test result"])

# seperate training data into x, y1(Genetic Disorder), and y2(Disorder Subclass)
train_x_df = train_df[["Patient Age",
                        "Genes in mother's side",
                        "Inherited from father",
                        "Maternal gene",
                        "Paternal gene",
                        "Blood cell count (mcL)",
                        "Folic acid details (peri-conceptional)",
                        "H/O serious maternal illness",
                        "Assisted conception IVF/ART",
                        "History of anomalies in previous pregnancies",
                        "No. of previous abortion",
                        "White Blood cell count (thousand per microliter)",
                        "Symptom 1",
                        "Symptom 2",
                        "Symptom 3",
                        "Symptom 4",
                        "Symptom 5",
                        "Status_Alive",
                        "Status_Deceased",
                        "Respiratory Rate (breaths/min)_Normal (30-60)",
                        "Respiratory Rate (breaths/min)_Tachypnea",
                        "Heart Rate (rates/min_Normal",
                        "Heart Rate (rates/min_Tachycardia",
                        "Follow-up_High",
                        "Follow-up_Low",
                        "Gender_Ambiguous",
                        "Gender_Female",
                        "Gender_Male",
                        "Birth defects_Multiple",
                        "Birth defects_Singular",
                        "Blood test result_abnormal",
                        "Blood test result_inconclusive",
                        "Blood test result_normal",
                        "Blood test result_slightly abnormal"]]

train_y1_df = train_df[["Genetic Disorder"]]

train_y2_df = train_df[["Disorder Subclass"]]

#encode categorical y1 into numerical labels 
enc1 = LabelEncoder()
enc1.fit_transform(train_y1_df)
train_y1_df = enc1.transform(train_y1_df).reshape(-1,1)

#encode categorical y2 into numerical labels 
enc2 = LabelEncoder()
enc2.fit_transform(train_y2_df)
train_y2_df = enc2.transform(train_y2_df).reshape(-1,1)

test_x_df = test_df

# split training and test data for each y
X1_train, X1_test, y1_train, y1_test = train_test_split(train_x_df, train_y1_df, stratify = train_y1_df, test_size=0.33, random_state=7)
X2_train, X2_test, y2_train, y2_test = train_test_split(train_x_df, train_y2_df, stratify = train_y2_df, test_size=0.33, random_state=7)

print()

class MyMLP(nn.Module):
    # Two-layer MLP (not really a perceptron activation function...) network class
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, C):
        super(MyMLP, self).__init__()
        # Fully connected layer WX + b mapping from input_dim (n) -> hidden_layer_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim1)
        self.hidden_fc1 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.hidden_fc2 = nn.Linear(hidden_dim2, hidden_dim3)
        # Output layer again fully connected mapping from hidden_layer_dim -> outputs_dim (C)
        self.output_fc = nn.Linear(hidden_dim2, C)
        
    # Don't call this function directly!! 
    # Simply pass input to model and forward(input) returns output, e.g. model(X)
    def forward(self, X):
        # X = [batch_size, input_dim (n)]
        X = self.input_fc(X)
        # Nonlinear activation function, e.g. ReLU (default good choice)
        # Could also choose F.softplus(x) for smooth-ReLU, empirically worse than ReLU
        X = F.tanh(X)
        X = self.hidden_fc1(X)
        X = F.tanh(X)
        # X = [batch_size, hidden_dim]
        # Connect to last layer and output 'logits'
        y = self.output_fc(X)
        return y

    
input_dim = X1_train.shape[1]
n_hidden_neurons1 = 16
n_hidden_neurons2 = 16
output_dim = 3

# It's called an MLP but really it's not...
model = MyMLP(input_dim, n_hidden_neurons1, n_hidden_neurons2, output_dim)
# Visualize network architecture
print(model)

def model_train(model, data, labels, criterion, optimizer, num_epochs=25):
    # Apparently good practice to set this "flag" too before training
    # Does things like make sure Dropout layers are active, gradients are updated, etc.
    # Probably not a big deal for our toy network, but still worth developing good practice
    model.train()
    # Optimize the neural network
    for epoch in range(num_epochs):
        # These outputs represent the model's predicted probabilities for each class. 
        outputs = model(data)
        # Criterion computes the cross entropy loss between input and target
        loss = criterion(outputs, labels)
        # Set gradient buffers to zero explicitly before backprop
        optimizer.zero_grad()
        # Backward pass to compute the gradients through the network
        loss.backward()
        # GD step update
        optimizer.step()
        
    return model


# Stochastic GD with learning rate and momentum hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.8, weight_decay=0)
# The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to 
# the output when validating, on top of calculating the negative log-likelihood using 
# nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
criterion = nn.CrossEntropyLoss()
num_epochs = 1000

X1_train = X1_train.astype('float64').values
y1_train = y1_train.ravel()

# Convert numpy structures to PyTorch tensors, as these are the data types required by the library
X_tensor = torch.FloatTensor(X1_train)
y_tensor = torch.LongTensor(y1_train)

# Trained model
model = model_train(model, X_tensor, y_tensor, criterion, optimizer, num_epochs=num_epochs)

def model_predict(model, data):
    # Similar idea to model.train(), set a flag to let network know your in "inference" mode
    model.eval()
    # Disabling gradient calculation is useful for inference, only forward pass!!
    with torch.no_grad():
        # Evaluate nn on test data and compare to true labels
        predicted_labels = model(data)
        # Back to numpy
        predicted_labels = predicted_labels.detach().numpy()

        return np.argmax(predicted_labels, 1)

X1_test = X1_test.astype('float64').values
y1_test = y1_test.ravel()

# Set up test data as tensor
X_test_tensor = torch.FloatTensor(X1_test)  
y_test_tensor = torch.LongTensor(y1_test) 
# Z matrix are the predictions resulting from the forward pass through the network
test1_pred = model_predict(model, X_test_tensor)

# print(Z)

unique, counts = np.unique(test1_pred, return_counts=True)
print(dict(zip(unique, counts)))

#check scores
score1 = f1_score(y_test_tensor, test1_pred, average='macro')

print(np.round(score1, 3))


# text confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(y_pred=test1_pred, y_true=y1_test)
print(conf_mat)

# figure confusion matrix
conf_display = ConfusionMatrixDisplay.from_predictions(y_pred=test1_pred, y_true=y1_test, colorbar=False)
plt.ylabel('Predicted Labels')
plt.xlabel('True Labels')
plt.title("Genetic Disorder")
plt.grid(False)
plt.show() 














# train_dataloader = DataLoader(train_data, batch_size=500, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64) # No need to shuffle...

# def model_train_loader(model, dataloader, criterion, optimizer):
#     size = len(dataloader.dataset)

#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         # Compute prediction error
#         predictions = model(X)
#         loss = criterion(predictions, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Report loss every 10 batches
#         if batch % 10 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def model_test_loader(model, dataloader, criterion):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     # Tracking test loss (cross-entropy) and correct classification rate (accuracy)
#     test_loss, correct = 0, 0
    
#     model.eval()
#     with torch.no_grad():
#         for X, y in dataloader:
#             predictions = model(X)
#             test_loss += criterion(predictions, y)
#             correct += (predictions.argmax(1) == y).type(torch.float).sum()
            
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

# # Let's train the Sequential model this time
# # And look at how we're training + testing in parallel
# # Useful if we wanted to do something like early stopping!
# # Nesterov is a better revision of the momentum update
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
# for t in range(num_epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     model_train_loader(model, train_dataloader, criterion, optimizer)
#     model_test_loader(model, test_dataloader, criterion)


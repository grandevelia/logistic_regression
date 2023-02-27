import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

#Plotting KNN
fn = f"data/D2z.txt"
df = pd.read_csv(fn, sep=" ", names=["x","y", "label"])

step = 0.1
start = -2
stop = 2 + step
grid_x = np.arange(start, stop, step)
grid_y = np.arange(start, stop, step)
xx, yy = np.meshgrid(grid_x, grid_y)
points = np.vstack([xx.ravel(), yy.ravel()]).transpose()

dists = euclidean_distances(points, df[["x", "y"]])
dist_min_idxs = np.argmin(dists, axis=1)
point_labels = df.loc[dist_min_idxs, "label"]
point_df = pd.DataFrame(points)
point_df.columns = ["x", "y"]
point_df["label"] = point_labels.reset_index(drop=True)

point_df["label"][point_df["label"] == 1] = "red"
point_df["label"][point_df["label"] == 0] = "blue"
df["label"][df["label"] == 1] = "+"
df["label"][df["label"] == 0] = "o"

plt.figure()
plt.scatter(point_df['x'], point_df['y'], color=point_df["label"], s=1)
plt.scatter(list(df['x']), list(df['y']), color="black", s=2)
out_fn = f"nn_classifier.png"
plt.savefig(out_fn, bbox_inches='tight')
plt.close()

#Spam filter
fn = "data/emails.csv"
df = pd.read_csv(fn)
feature_cols = df.columns[~df.columns.isin(["Email No.", "Prediction"])]

#5-fold cross validation splits
step = df.shape[0]/5
cv_splits = np.arange(0, df.shape[0] + step, step)

def knn(k, model, input, feature_cols, label_col="label", predict_mode="MODE"):
    '''
    model and input should have corresponding shapes
    '''
    dists = euclidean_distances(input[feature_cols], model[feature_cols])
    if k == 1:
        dist_min_idxs = np.argmin(dists, axis=1)
        point_labels = model.loc[dist_min_idxs, label_col].reset_index(drop=True)
    else:
        dist_min_idxs = np.argpartition(dists, k, axis=1)[:, :k]
        point_labels = np.array(model[label_col])[dist_min_idxs]
        if predict_mode == "MODE":
            #Use most common prediction
            point_labels = np.array(pd.DataFrame(point_labels).T.mode().T).squeeze()
        else:
            #Report confidence
            point_labels = np.array(pd.DataFrame(point_labels).T.mean().T).squeeze()
    input[label_col] = point_labels
    return input


def accuracy(pred, true):
    pred, true = np.array(pred), np.array(true)
    return (pred == true).sum()/pred.shape[0]


def precision(pred, true):
    pred, true = np.array(pred), np.array(true)
    tp = (pred * true).sum()
    fp = (pred * np.abs(true-1)).sum()
    return round(tp/(tp + fp), 3)


def recall(pred, true):
    pred, true = np.array(pred), np.array(true)
    tp = (pred * true).sum()
    fn = (np.abs(pred-1) * true).sum()
    return round(tp/(tp + fn), 3)


def fpr(pred, true):
    pred, true = np.array(pred), np.array(true)
    fp = (pred * np.abs(true-1)).sum()
    tn = (np.abs(pred-1) * np.abs(pred-1)).sum()
    return round(fp/(fp + tn), 3)


def get_splits(i, df, cv_splits, print_metrics=True):
    train_start = cv_splits[i]
    train_stop = cv_splits[i+1]
    if print_metrics:
        print(f"Evaluating rows {int(train_start)} to {int(train_stop)}")
    test_rows = df.loc[train_start:train_stop-1]
    if train_start == 0:
        train_rows = df.loc[train_stop:]
    else:
        train_rows = pd.concat([df.loc[0:train_start-1], df.loc[train_stop:]])
    return train_rows, test_rows


def get_metrics(preds, true, print_metrics):
    pred_accuracy = accuracy(preds, true)
    pred_precision = precision(preds, true)
    pred_recall = recall(preds, true)
    if print_metrics:
        print(f"Accuracy: {pred_accuracy}, Precision: {pred_precision}, Recall: {pred_recall}") 
    return pred_accuracy, pred_precision, pred_recall

            
def run_knn(k, df, feature_cols, cv_splits, print_metrics=True):
    accuracies = []
    for i in range(len(cv_splits) - 1):
        train_rows, test_rows = get_splits(i, df, cv_splits, print_metrics)
        preds = knn(k, train_rows.reset_index(drop=True), test_rows.reset_index(drop=True), feature_cols, "Prediction")
        pred_accuracy, pred_precision, pred_recall = get_metrics(preds["Prediction"], test_rows["Prediction"], print_metrics)
        accuracies += [pred_accuracy]
    return np.array(accuracies).mean()


#5-fold cross validation, KNN k=1
_ = run_knn(1, df, feature_cols, cv_splits)

#5-fold cross validation, KNN, multiple ks
ks = [1, 3, 5, 7, 10]
accuracies = []
for k in ks:
    accuracies += [run_knn(k, df, feature_cols, cv_splits, print_metrics=False)]

plt.figure()
plt.plot(ks, accuracies)
plt.savefig("5fold_knn_accuracies.png")
plt.close()

#Logistic Regression

def sigmoid(x):
    return np.piecewise(
        x,
        [x > 0],
        [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
    )

def del_cross_entropy_loss(preds, target):
    '''
    preds should be sigmoid(input)
    '''
    return preds - target

class LogisticRegression():
    def __init__(self, input_data, step_size=0.1, pred_col="label", feature_cols=["x", "y"], n_steps=50):
        self.pred_col = pred_col
        self.feature_cols = feature_cols
        self.targets = np.array(input_data.loc[:, pred_col]).squeeze()
        self.input_data = input_data.loc[:, feature_cols].assign(bias=1) #Add bias term (1 is neutral)
        self.params = np.zeros([len(feature_cols)+1])
        self.step_size = step_size
        self.n_steps = n_steps
        
    def fit(self):
        for _ in range(self.n_steps):
            preds = sigmoid(np.dot(self.input_data, self.params)) #n data points
            grad = del_cross_entropy_loss(preds, self.targets) #n data points
            feature_grad = np.dot(self.input_data.T, grad) #n features
            next_step = self.params - (self.step_size * feature_grad)/self.input_data.shape[0]
            self.params = next_step
    
    def inference(self, test_data):
        test_data = test_data.loc[:, self.feature_cols].assign(bias=1)
        return sigmoid(np.dot(test_data, self.params))

def run_logistic_regression(df, feature_cols, cv_splits, print_metrics=True):
    accuracies = []
    for i in range(len(cv_splits) - 1):
        train_rows, test_rows = get_splits(i, df, cv_splits, print_metrics)
        lr = LogisticRegression(train_rows.reset_index(drop=True), step_size=0.25, pred_col="Prediction", feature_cols=feature_cols)
        lr.fit()
        preds = lr.inference(test_rows.reset_index(drop=True))
        pred_accuracy, pred_precision, pred_recall = get_metrics(preds, test_rows["Prediction"], print_metrics)
        accuracies += [pred_accuracy]
    return np.array(accuracies).mean()

logistic_accuracies = run_logistic_regression(df, feature_cols, cv_splits)

#ROC curves for KNN=5 vs Logistic Regression

single_train_split = 4000

knn_tprs = []
knn_fprs = []
for boundary in range(0, 120, 20):
    boundary = boundary/100
    single_train_split = 4000
    single_train_rows = df[:single_train_split].reset_index(drop=True).copy()
    single_test_rows = df[single_train_split:].reset_index(drop=True).copy()
    knn_preds = knn(5, single_train_rows, single_test_rows, feature_cols, "Prediction", predict_mode="MEAN")["Prediction"].copy()
    if boundary == 0:
        knn_preds.loc[knn_preds >= boundary] = 1.0
    else:
        knn_preds.loc[knn_preds > boundary] = 1.0
    knn_preds.loc[knn_preds <= boundary] = 0.0
    single_test_rows = df[single_train_split:].reset_index(drop=True).copy()
    knn_tprs += [recall(knn_preds, single_test_rows["Prediction"])]
    knn_fprs += [fpr(knn_preds, single_test_rows["Prediction"])]

single_train_rows = df[:single_train_split].reset_index(drop=True).copy()
single_test_rows = df[single_train_split:].reset_index(drop=True).copy()

lr = LogisticRegression(single_train_rows, step_size=0.01, n_steps=750, pred_col="Prediction", feature_cols=feature_cols)
lr.fit()
lr_preds = lr.inference(single_test_rows)

logistic_tprs = []
logistic_fprs = []
for boundary in range(0, 1001, 1):
    boundary = boundary/1000
    boundary_preds = lr_preds.copy()
    if boundary == 0:
        boundary_preds[boundary_preds >= boundary] = 1.0
    else:
        boundary_preds[boundary_preds > boundary] = 1.0
    boundary_preds[boundary_preds <= boundary] = 0.0
    logistic_tprs += [recall(boundary_preds, single_test_rows["Prediction"])]
    logistic_fprs += [fpr(boundary_preds, single_test_rows["Prediction"])]

plt.figure()
plt.plot(logistic_fprs, logistic_tprs)
plt.plot(knn_fprs, knn_tprs)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(['Logistic', 'KNN'])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("knn_logistic_ROC.test.png", bbox_inches='tight')
plt.close()

#Question 5 ROC

sample_df = pd.DataFrame({"confidence": [0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1],
                          "label": [1, 1, 0, 1, 1, 0, 1, 1, 0, 0]})

thresh_tprs = []
thresh_fprs = []
for thresh in range(0, 11, 1):
    thresh = thresh/10
    thresh_df = sample_df.copy()
    thresh_df.loc[:, "confidence"][thresh_df["confidence"] > thresh] = 1
    thresh_df.loc[:, "confidence"][thresh_df["confidence"] != 1] = 0
    thresh_tprs += [recall(thresh_df["confidence"], thresh_df["label"])]
    thresh_fprs += [fpr(thresh_df["confidence"], thresh_df["label"])]

plt.figure()
plt.plot(thresh_fprs, thresh_tprs)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("q5_ROC.png", bbox_inches='tight')
plt.close()
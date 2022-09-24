#%%
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from pytictoc import TicToc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import svm
#%%
# Churn data
scoring = {'recall': make_scorer(recall_score)}
churn = pd.read_csv('https://raw.githubusercontent.com/brandonritchie/CS7641Assignments/main/Bank%20Customer%20Churn%20Prediction.csv')

# %%
# Data cleaning
cleaned_churn = churn.drop(['customer_id'], axis = 1)

# Encode for models later on
churn = pd.get_dummies(cleaned_churn, columns = ['country', 'gender'])

#%%
# Check how balanced the data is
plt_dat = pd.DataFrame(churn.value_counts('churn')).reset_index().replace({'churn':{0:'False', 1:'True'}})
plt.bar(plt_dat.churn,plt_dat[0])
plt.title('Distribution of Prediction: Customer Left Bank or Not')
plt.show()
# We will focus on the recall performance metric when evaluating the models since it will likely try to overpredict no churn.
# From a business standpoint, we would want to minimize the number of people that we would predict won't churn because they will not be recognized
# and targeted accordingly
# %%
# Split data
X = churn.drop(['churn'], axis = 1)
y = churn[['churn']]

X_train1,X_test1,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train2, X_val1, y_train, y_val = train_test_split(X_train1,y_train,test_size=0.2)
y_train = y_train.reset_index().drop('index', axis = 1)
y_test = y_test.reset_index().drop('index', axis = 1)
sc_train = StandardScaler()
sc_val = StandardScaler()
sc_test = StandardScaler()
X_train = pd.DataFrame(sc_train.fit_transform(X_train2.values), columns = X_train2.columns)
X_val = pd.DataFrame(sc_train.fit_transform(X_val1.values), columns = X_val1.columns)
X_test = pd.DataFrame(sc_test.fit_transform(X_test1.values), columns = X_test1.columns)

def calculate_precision_recall(real, pred):
    real = list(real)
    pred = list(pred)
    md = pd.DataFrame({'Real':real, 'Pred':pred})
    md['TP'] = np.where((md['Real'] == 1) & (md['Real'] == md['Pred']), 1 , 0)
    md['FN'] = np.where((md['Real'] == 1) & (md['Pred'] == 0), 1 , 0)
    md['FP'] = np.where((md['Real'] == 0) & (md['Pred'] == 1), 1 , 0)
    recall = sum(md['TP']) / (sum(md['TP']) + sum(md['FN']))
    precision = sum(md['TP']) / (sum(md['TP']) + sum(md['FP']))
    return(recall, precision)
#%%
# HYPERPARAMETERS KNN
k_range = range(1,60,2)
scores_val_k = []
scores_train_k = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    # Cross Validation
    scores = cross_val_score(knn, X_train, y_train, cv = 5, scoring = 'recall')
    # Training data
    knn.fit(X_train, y_train)
    recall_val, precision_val = calculate_precision_recall(y_val.churn, knn.predict(X_val))
    scores_train_k.append(scores.mean())
    scores_val_k.append(recall_val)

p_range = range(1,60,2)
scores_val_k2 = []
scores_train_k2 = []

for p in p_range:
    knn = KNeighborsClassifier(p = p)
    # Cross Validation
    scores = cross_val_score(knn, X_train, y_train, cv = 5, scoring = 'precision')
    # Training data
    knn.fit(X_train, y_train)
    recall_val, precision_val = calculate_precision_recall(y_val.churn, knn.predict(X_val))
    scores_train_k2.append(scores.mean())
    scores_val_k2.append(precision_val)

plt.plot(k_range, scores_train_k, label = "Train")
plt.plot(k_range, scores_val_k, label = "Validation")
plt.xlabel('Number of Neighbors')
plt.ylabel('Recall')
plt.title('Recall by Number of Neighbors KNN - Churn')
plt.legend()
plt.show()

plt.plot(k_range, scores_train_k2, label = "Train")
plt.plot(k_range, scores_val_k2, label = "Validation")
plt.xlabel('Number of Neighbors')
plt.ylabel('Precision')
plt.title('Precision by Number of Neighbors KNN - Churn')
plt.legend()
plt.show()

#%%
#Learning Curve KNN
# https://stackoverflow.com/questions/54621429/what-does-the-learning-curve-in-classification-decision-tree-mean
# As the sample size increases for both the recall for the training and testing data increase.
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=None,scoring = scoring['recall'], n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Recall")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title("Learning Curve KNN - Churn")
    ax.legend(loc="best")

    return plt

fig = plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = KNeighborsClassifier(n_neighbors= 2, p = 1)
plot_learning_curve(estimator, X_train, y_train, ax = fig, train_sizes=np.linspace(.1, 1.0, 5))

# %%
# HYPERPARAMETERS Decision Tree
min_sample_leaves = range(1,100)
scores_train_ms = []
scores_val_ms = []

for i in min_sample_leaves:
    print(i)
    clf = tree.DecisionTreeClassifier(min_samples_leaf = i)
    # Training data
    clf.fit(X_train, y_train)
    # Cross Validation
    recall, precision = calculate_precision_recall(y_train.churn, clf.predict(X_train))
    recall_val, precision_val = calculate_precision_recall(y_val.churn, clf.predict(X_val))
    scores_train_ms.append(recall)
    scores_val_ms.append(recall_val)


max_depth = range(1,40)
scores_train_md = []
scores_val_md = []

for i in max_depth:
    print(i)
    clf = tree.DecisionTreeClassifier(max_depth = i)
    # Training data
    clf.fit(X_train, y_train)
    # Cross Validation
    scores = precision_recall_fscore_support(y_train, clf.predict(X_train), average = 'macro')[1]
    score_val = precision_recall_fscore_support(y_val, clf.predict(X_val), average = 'macro')[1]
    scores_train_md.append(scores.mean())
    scores_val_md.append(score_val)

plt.plot(min_sample_leaves, scores_train_ms, label = "Train")
plt.plot(min_sample_leaves, scores_val_ms, label = "Validation")
plt.xlabel('Minimum Sample Leaves')
plt.ylabel('Recall')
plt.title('Recall by Minimum Sample Leaves Decision Tree - Churn')
plt.legend()
plt.show()

plt.plot(max_depth, scores_train_md, label = "Train")
plt.plot(max_depth, scores_val_md, label = "Validation")
plt.xlabel('Max Depth')
plt.ylabel('Recall')
plt.title('Recall by Max Tree Depth Decision Tree - Churn')
plt.legend()
plt.show()

#%%
#Learning Curve
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv,scoring = scoring['recall'], n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Recall")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title("Learning Curve Decision Tree - Churn")
    ax.legend(loc="best")

    return plt

fig = plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = tree.DecisionTreeClassifier(min_samples_leaf = 13, max_depth = 7)
plot_learning_curve(estimator, X_train, y_train, ax = fig, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

plt.show()
# %%
# Hyperparamaters ANN

def build_ann_model(layers, max_nodes, layer_node_shape):
    prec = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    model = Sequential()
    for l in range(layers):
        if layer_node_shape == 'triangle':
            if l == 0:
                nodes = (((l + layers) / layers) * max_nodes) // 1
                model.add(Dense(nodes, input_dim=13, activation='relu'))
                model.add(Dropout(0.1))
            else:
                nodes = ((1-(((l + layers) / layers)) % 1) * max_nodes) // 1
                model.add(Dense(nodes))
                model.add(Dropout(0.1))

        if layer_node_shape == 'uniform':
            if l == 0:
                model.add(Dense(max_nodes, input_dim=13, activation='relu'))
                model.add(Dropout(0.1))
            else:
                model.add(Dense(max_nodes))
                model.add(Dropout(0.1))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))  
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=[recall, prec])

    return(model)

layers = [1,2,3,4]
max_nodes = [8,16,32,64,128]
shape = ['triangle', 'uniform']

l_list = []
mn_list = []
s_list = []
hist_l = []

for l in layers:
    for mn in max_nodes:
        for s in shape:
            model = build_ann_model(l, mn, s)
            history = model.fit(X_train, y_train ,verbose=1, epochs=100, batch_size=64,
                    validation_data=(X_val, y_val))
            hist_l.append(history)
            l_list.append(l)
            mn_list.append(mn)
            s_list.append(s)

recall_train_l = []
precision_train_l = []
recall_val_l = []
precision_val_l = []

for h in hist_l:
    recall = np.array(next(v for k,v in h.history.items() if 'recall_' in k)).mean()
    precision = np.array(next(v for k,v in h.history.items() if 'precision_' in k)).mean()
    recall_val = np.array(next(v for k,v in h.history.items() if 'val_recall' in k)).mean()
    precision_val = np.array(next(v for k,v in h.history.items() if 'val_precision' in k)).mean()
    recall_train_l.append(recall)
    precision_train_l.append(precision)
    recall_val_l.append(recall_val)
    precision_val_l.append(precision_val)

ann_df_churn = pd.DataFrame({
    'Layers':l_list,
    'Max_Nodes':mn_list,
    'Shape':s_list,
    'Recall_Train':recall_train_l,
    'Precision_Train':precision_train_l,
    'Recall_Validation':recall_val_l,
    'Precision_Validation':precision_val_l
})
ann_df_churn.to_csv('ann_churn.csv')
ann_df_churn = pd.read_csv('ann_churn.csv')
ann_df1 = ann_df_churn.query('Shape == "uniform"')[['Layers', 'Max_Nodes', 'Recall_Train']]
ann_df2 = ann_df_churn.query('Shape == "uniform"')[['Layers', 'Max_Nodes', 'Recall_Validation']]

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
fig.set_figheight(8)
fig.set_figwidth(8)
for label, df in ann_df1.groupby('Max_Nodes'):
    df.plot('Layers','Recall_Train', ax=ax1, label=label)
ax1.set_title('ANN Recall Performance Train - Churn')
ax1.get_legend().remove()
fig.legend(title="Max Nodes")
for label, df in ann_df2.groupby('Max_Nodes'):
    df.plot('Layers','Recall_Validation', ax=ax2, label=label)
ax2.set_title('ANN Recall Performance Validation - Churn') 
ax2.get_legend().remove() 

#%%
# Loss curve
model = build_ann_model(3, 128, 'uniform')
history = model.fit(X_train, y_train ,verbose=1, epochs=100, batch_size=32,
                    validation_data=(X_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss - Churn')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = next(v for k,v in history.history.items() if 'recall' in k)
val_acc = next(v for k,v in history.history.items() if 'val_recall' in k)
plt.plot(epochs, acc, 'y', label='Training Recall')
plt.plot(epochs, val_acc, 'r', label='Validation Recall')
plt.title('Training and validation recall - Churn')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.show()

#%%
#Learning Curve ANN
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv,scoring = scoring['recall'], n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Recall")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title("Learning Curve Neural Network - Churn")
    ax.legend(loc="best")

    return plt

fig = plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = MLPClassifier(hidden_layer_sizes=(3,128), activation = 'logistic', solver = 'adam')
plot_learning_curve(estimator, X_train, y_train, ax = fig, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

# %%
# AdaBoost Model
e_range = [10, 20,30,40,50,60,70,80]
md_range = [1,2,3,4]
scores_val_e = []
scores_train_e = []
e_l = []
md_l = []

for e in e_range:
    for mdv in md_range:
        base = tree.DecisionTreeClassifier(max_depth= mdv)
        ada_model = AdaBoostClassifier(base_estimator=base, n_estimators = e)
        # Cross Validation
        scores = cross_val_score(ada_model, X_train, y_train, cv = 5, scoring = 'recall')
        # Training data
        ada_model.fit(X_train, y_train)

        recall, precision = calculate_precision_recall(y_val.churn, ada_model.predict(X_val))

        score_val = recall
        scores_val_e.append(score_val)
        scores_train_e.append(scores.mean())
        e_l.append(e)
        md_l.append(mdv)

boost_dat = pd.DataFrame({'Validation_Metric':scores_val_e,'Train_Metric':scores_train_e,'Num_Estimator':e_l,'Max_Depth':md_l})
boost_dat_v = boost_dat[['Validation_Metric', 'Num_Estimator', 'Max_Depth']]
boost_dat_t = boost_dat[['Train_Metric', 'Num_Estimator', 'Max_Depth']]

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
fig.set_figheight(8)
fig.set_figwidth(8)
for label, df in boost_dat_t.groupby('Max_Depth'):
    df.plot('Num_Estimator','Train_Metric', ax=ax1, label=label)
ax1.set_title('AdaBoost Recall Performance Train - Churn')
fig.legend(title="Max Depth")
for label, df in boost_dat_v.groupby('Max_Depth'):
    df.plot('Num_Estimator','Validation_Metric', ax=ax2, label=label)
ax2.set_title('AdaBoost Recall Performance Validation - Churn')

#%%
#Learning Curve Adaboost
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,scoring = scoring['recall'], n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Recall")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title("Learning Curve AdaBoost - Churn")
    ax.legend(loc="best")

    return plt

fig = plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
base = tree.DecisionTreeClassifier(max_depth= 3)
ada_model = AdaBoostClassifier(base_estimator=base, n_estimators = 20)
plot_learning_curve(ada_model, X_train, y_train, ax = fig, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

#%%
# SVM Model
from sklearn.model_selection import validation_curve
kernels = ['poly', 'rbf', 'sigmoid']

degree_range = np.array([1,2,3,4,5,6,7])
train_scores, valid_scores = validation_curve(
svm.SVC(kernel='poly'), X_train, y_train, param_name="degree", param_range=degree_range,
cv=5, scoring = 'recall')
train_scores_mean = np.mean(train_scores, axis=1)
validation_scores_mean = np.mean(valid_scores, axis=1)

plt.plot(degree_range, train_scores_mean, label = 'Training Data')
plt.plot(degree_range, validation_scores_mean, label = 'Validation Data')
plt.xlabel('Polynomial Degrees')
plt.ylabel('Recall Score')
plt.title('Poly Kernel Type Optimal Degrees - Churn')
plt.legend(loc='best')

# Compare Kernels
recall_score = []
recall_score_val = []
for k in kernels:
    if k == 'poly':
        svm_model = svm.SVC(kernel = 'poly', degree = 3)
    else:
        svm_model = svm.SVC(kernel = k)
    
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_val)
    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    scores = cross_val_score(svm_model, X_train, y_train, scoring='recall', cv=cv, n_jobs=-1)
    recall, precision = calculate_precision_recall(y_val.churn, y_pred)
    recall_score.append(scores.mean())
    recall_score_val.append(recall)

N = 3
ind = np.arange(N)
width = 0.35
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, recall_score, 0.35, color='royalblue', label = 'Training Data')
rects2 = ax.bar(ind + 0.35, recall_score_val, 0.35, color='seagreen', label = 'Validation Data')
plt.ylabel('Recall Score')
plt.title('SVM Kernel Types by Recall Score - Churn')
plt.xticks(ind + width / 2, ('Poly (3 Degree)', 'RBF', 'Sigmoid'))
# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()


gamma_range = np.arange(0.01,0.3,0.02)
train_scores, valid_scores = validation_curve(
svm.SVC(kernel='poly'), X_train, y_train, param_name="gamma", param_range=gamma_range,
cv=5, scoring = 'recall')
train_scores_mean = np.mean(train_scores, axis=1)
validation_scores_mean = np.mean(valid_scores, axis=1)

plt.plot(gamma_range, train_scores_mean, label = 'Training Data')
plt.plot(gamma_range, validation_scores_mean, label = 'Validation Data')
plt.xlabel('Gamma')
plt.ylabel('Recall Score')
plt.title('Optimal Gamma SVM - Churn')
plt.legend(loc='best')
# %%
# Learning Curve SVM
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,scoring = 'accuracy', n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Accuracy")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title("Learning Curve SVM - Churn")
    ax.legend(loc="best")

    return plt

fig = plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
model = svm.SVC(kernel = 'rbf', gamma = 0.1)
plot_learning_curve(model, X_train, y_train, ax = fig, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

# %%
# Wall Clock times
# Decision Tree Model
dt_model = tree.DecisionTreeClassifier(min_samples_leaf = 13, max_depth = 7)
# KNN Model
knn_model = KNeighborsClassifier(n_neighbors= 15)
# AdaBoost Model
base = tree.DecisionTreeClassifier(max_depth= 3)
ada_model = AdaBoostClassifier(base_estimator=base, n_estimators = 20)
# ANN model
ann_model = MLPClassifier(hidden_layer_sizes=(3,128), activation = 'logistic', solver = 'adam')
# SVM Model
svm_model = svm.SVC(kernel = 'poly', degree = 3, gamma = 0.1)

models = {'Decision Tree' : dt_model, 'KNN' : knn_model, 'AdaBoost' : ada_model, 'ANN' : ann_model, 'SVM' : svm_model}


t = TicToc()
model_fit = []
model_query = []
test_recall = []
test_precision = []
model_l = []
for i, (k,v) in enumerate(models.items()):
    # Fit model time
    t.tic()
    v.fit(X_train, y_train)
    model_fit.append(t.tocvalue())
    t.toc()

    # Query time
    t.tic()
    y_pred = v.predict(X_test)
    model_query.append(t.tocvalue())
    t.toc()

    recall, precision = calculate_precision_recall(y_test.churn, y_pred)
    test_recall.append(recall)
    test_precision.append(precision)
    model_l.append(k)

wc = pd.DataFrame({'model':model_l,
'fit_time':model_fit,
'query_time':model_query,
'recall':test_recall,
'precision':test_precision})

wc2 = wc.sort_values(by = ['fit_time'], ascending = False)
wc3 = wc.sort_values(by = ['recall'], ascending = False)

N = 5
ind = np.arange(N)
width = 0.35
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, wc2.fit_time, 0.35, label = 'Fit Time')
rects2 = ax.bar(ind + 0.35, wc2.query_time, 0.35, label = 'Query Time')
plt.ylabel('Seconds')
plt.title('Fit and Query Times by Model - Churn')
plt.xticks(ind + width / 2, ('ANN', 'SVM', 'AdaBoost','KNN','Decision Tree'))
# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

N = 5
ind = np.arange(N)
width = 0.35
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, wc3.recall, 0.35, label = 'Recall')
rects2 = ax.bar(ind + 0.35, wc3.precision, 0.35, label = 'Precision')
plt.ylabel('Scoring')
plt.title('Optimized Model Performances on Test Data - Churn')
plt.xticks(ind + width / 2, ('AdaBoost', 'ANN', 'DecisionTree','SVM','KNN'))
# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()
# %%

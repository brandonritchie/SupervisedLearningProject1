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
# Product subscription
data_raw = pd.read_csv('https://raw.githubusercontent.com/brandonritchie/CS7641Assignments/main/product_subscription.csv')

# Check how balanced the data is
plt.hist(data_raw.term_deposit_subscribed)
plt_dat = pd.DataFrame(data_raw.value_counts('term_deposit_subscribed')).reset_index().replace({'term_deposit_subscribed':{0:'False', 1:'True'}})
plt.bar(plt_dat.term_deposit_subscribed,plt_dat[0])
plt.title('Distribution of Prediction: Subscription to Term Deposit or Not')
plt.show()

#%%
# Data Cleaning
# One hot encoding
data_raw = pd.get_dummies(data_raw, columns = ['job_type','marital','education','default', 'prev_campaign_outcome'])

# Transformation of date time: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
data_raw['sin_time'] = np.sin(2*np.pi*data_raw.day_of_month/365)
data_raw['cos_time'] = np.cos(2*np.pi*data_raw.day_of_month/365)

data_raw['housing_loan'] = np.where(data_raw['housing_loan'] == 'yes', 1,0)
data_raw['personal_loan'] = np.where(data_raw['personal_loan'] == 'yes', 1,0)

data_raw = data_raw.fillna(data_raw.mean())

data_cleaned = data_raw.drop(columns=['communication_type', 'day_of_month', 'month', 'id','days_since_prev_campaign_contact'])

# Downsample to balanced data
sub = data_cleaned.loc[data_cleaned.term_deposit_subscribed == 1]
nsub = data_cleaned.loc[data_cleaned.term_deposit_subscribed == 0].head(len(sub))
data_cleaned = pd.concat([sub,nsub])
# %%
# Split data
X = data_cleaned.drop(['term_deposit_subscribed'], axis = 1)
y = data_cleaned[['term_deposit_subscribed']]

X_train1,X_test1,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train2, X_val1, y_train, y_val = train_test_split(X_train1,y_train,test_size=0.3)
y_train = y_train.reset_index().drop('index', axis = 1)
y_test = y_test.reset_index().drop('index', axis = 1)
sc_train = StandardScaler()
sc_val = StandardScaler()
sc_test = StandardScaler()
X_train = pd.DataFrame(sc_train.fit_transform(X_train2.values), columns = X_train2.columns)
X_val = pd.DataFrame(sc_train.fit_transform(X_val1.values), columns = X_val1.columns)
X_test = pd.DataFrame(sc_test.fit_transform(X_test1.values), columns = X_test1.columns)

def calculate_precision_recall_accuracy(real, pred):
    real = list(real)
    pred = list(pred)
    md = pd.DataFrame({'Real':real, 'Pred':pred})
    md['TP'] = np.where((md['Real'] == 1) & (md['Real'] == md['Pred']), 1 , 0)
    md['TN'] = np.where((md['Real'] == 0) & (md['Real'] == md['Pred']), 1 , 0)
    md['FN'] = np.where((md['Real'] == 1) & (md['Pred'] == 0), 1 , 0)
    md['FP'] = np.where((md['Real'] == 0) & (md['Pred'] == 1), 1 , 0)
    recall = sum(md['TP']) / (sum(md['TP']) + sum(md['FN']))
    precision = sum(md['TP']) / (sum(md['TP']) + sum(md['FP']))
    accuracy = (sum(md['TP']) + sum(md['TN'])) / len(md)
    return(recall, precision, accuracy)
#%%
# HYPERPARAMETERS KNN
k_range = range(1,100,10)
scores_val_k = []
scores_train_k = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    # Cross Validation
    scores = cross_val_score(knn, X_train, y_train, cv = 5, scoring = 'recall')
    # Training data
    knn.fit(X_train, y_train)
    recall_val, precision_val, accuracy_val = calculate_precision_recall_accuracy(y_val.term_deposit_subscribed, knn.predict(X_val))
    scores_train_k.append(scores.mean())
    scores_val_k.append(recall_val)

k_range2 = range(1,100,5)
scores_val_k2 = []
scores_train_k2 = []

for k in k_range2:
    knn = KNeighborsClassifier(n_neighbors = k)
    # Cross Validation
    scores = cross_val_score(knn, X_train, y_train, cv = 5, scoring = 'accuracy')
    # Training data
    knn.fit(X_train, y_train)
    recall_val, precision_val, accuracy_val = calculate_precision_recall_accuracy(y_val.term_deposit_subscribed, knn.predict(X_val))
    scores_train_k2.append(scores.mean())
    scores_val_k2.append(accuracy_val)

plt.plot(k_range, scores_train_k, label = "Train")
plt.plot(k_range, scores_val_k, label = "Validation")
plt.xlabel('Number of Neighbors')
plt.ylabel('Recall')
plt.title('Recall by Number of Neighbors KNN - Product Subscription')
plt.legend()
plt.show()

plt.plot(k_range2, scores_train_k2, label = "Train")
plt.plot(k_range2, scores_val_k2, label = "Validation")
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy by Number of Neighbors KNN - Product Subscription')
plt.legend()
plt.show()

#%%
#Learning Curve KNN
scoring = {'recall': make_scorer(recall_score)}
# https://stackoverflow.com/questions/54621429/what-does-the-learning-curve-in-classification-decision-tree-mean
# As the sample size increases for both the recall for the training and testing data increase.
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=None,scoring = 'accuracy', n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Accuracy")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title("Learning Curve KNN - Product Subscription")
    ax.legend(loc="best")

    return plt

fig = plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = KNeighborsClassifier(n_neighbors= 18, p = 1)
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
    recall, precision, accuracy = calculate_precision_recall_accuracy(y_train.term_deposit_subscribed, clf.predict(X_train))
    recall_val, precision_val, accuracy_val = calculate_precision_recall_accuracy(y_val.term_deposit_subscribed, clf.predict(X_val))
    scores_train_ms.append(accuracy)
    scores_val_ms.append(accuracy_val)


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
plt.ylabel('Accuracy')
plt.title('Accuracy by Min Sample Leaves Decision Tree - Product Subscription')
plt.legend()
plt.show()

plt.plot(max_depth, scores_train_md, label = "Train")
plt.plot(max_depth, scores_val_md, label = "Validation")
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy by Max Depth Decision Tree - Product Subscription')
plt.legend()
plt.show()
# %%
#Learning Curve
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv,scoring = 'accuracy', n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Accuracy")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title('Learning Curve Decision Tree - Product Subscription')
    ax.legend(loc="best")

    return plt

fig = plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = tree.DecisionTreeClassifier(min_samples_leaf = 55, max_depth = 5)
plot_learning_curve(estimator, X_train, y_train, ax = fig, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

plt.show()
# %%
# Hyperparamaters ANN
def build_ann_model(layers, max_nodes, layer_node_shape):
    prec = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    accuracy = keras.metrics.Accuracy()
    model = Sequential()
    for l in range(layers):
        if layer_node_shape == 'triangle':
            if l == 0:
                nodes = (((l + layers) / layers) * max_nodes) // 1
                model.add(Dense(nodes, input_dim=34, activation='relu'))
                model.add(Dropout(0.1))
            else:
                nodes = ((1-(((l + layers) / layers)) % 1) * max_nodes) // 1
                model.add(Dense(nodes))
                model.add(Dropout(0.1))

        if layer_node_shape == 'uniform':
            if l == 0:
                model.add(Dense(max_nodes, input_dim=34, activation='relu'))
                model.add(Dropout(0.1))
            else:
                model.add(Dense(max_nodes))
                model.add(Dropout(0.1))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))  
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=[recall, prec, 'binary_accuracy'])

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
accuracy_train_l = []
precision_train_l = []
recall_val_l = []
precision_val_l = []
accuracy_val_l = []

for h in hist_l:
    recall = np.array(next(v for k,v in h.history.items() if 'recall_' in k)).mean()
    accuracy = np.array(next(v for k,v in h.history.items() if 'binary_accuracy' in k)).mean()
    precision = np.array(next(v for k,v in h.history.items() if 'precision_' in k)).mean()
    recall_val = np.array(next(v for k,v in h.history.items() if 'val_recall' in k)).mean()
    accuracy_val = np.array(next(v for k,v in h.history.items() if 'val_binary_accuracy' in k)).mean()
    precision_val = np.array(next(v for k,v in h.history.items() if 'val_precision' in k)).mean()
    recall_train_l.append(recall)
    accuracy_train_l.append(accuracy)
    precision_train_l.append(precision)
    recall_val_l.append(recall_val)
    accuracy_val_l.append(accuracy_val)
    precision_val_l.append(precision_val)

ann_df_sub = pd.DataFrame({
    'Layers':l_list,
    'Max_Nodes':mn_list,
    'Shape':s_list,
    'Recall_Train':recall_train_l,
    'Accuracy_Train':accuracy_train_l,
    'Precision_Train':precision_train_l,
    'Recall_Validation':recall_val_l,
    'Accuracy_Validation':accuracy_val_l,
    'Precision_Validation':precision_val_l
})
ann_df_sub.to_csv('ann_sub.csv')
ann_df_sub = pd.read_csv('ann_sub.csv')
ann_df1 = ann_df_sub.query('Shape == "uniform"')[['Layers', 'Max_Nodes', 'Accuracy_Train']]
ann_df2 = ann_df_sub.query('Shape == "uniform"')[['Layers', 'Max_Nodes', 'Accuracy_Validation']]

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
fig.set_figheight(8)
fig.set_figwidth(8)
for label, df in ann_df1.groupby('Max_Nodes'):
    df.plot('Layers','Accuracy_Train', ax=ax1, label=label)
ax1.set_title('ANN Accuracy Performance Train - Product Subscription')
ax1.get_legend().remove()
fig.legend(title="Max Nodes")
for label, df in ann_df2.groupby('Max_Nodes'):
    df.plot('Layers','Accuracy_Validation', ax=ax2, label=label)
ax2.set_title('ANN Accuracy Performance Validation - Product Subscription')
ax2.get_legend().remove()
# %%
# Loss curve
model = build_ann_model(3, 128, 'uniform')
history = model.fit(X_train, y_train ,verbose=1, epochs=100, batch_size=64,
                    validation_data=(X_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss - Product Subscription')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = next(v for k,v in history.history.items() if 'binary_accuracy' in k)
val_acc = next(v for k,v in history.history.items() if 'val_binary_accuracy' in k)
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy - Product Subscription')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%%
#Learning Curve ANN
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv,scoring = 'accuracy', n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Accuracy")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title('Learning ANN - Product Subscription')
    ax.legend(loc="best")

    return plt

fig = plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = MLPClassifier(hidden_layer_sizes=(3,128), activation = 'logistic', solver = 'adam', max_iter = 10)
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

        recall, precision, accuracy = calculate_precision_recall_accuracy(y_val.term_deposit_subscribed, ada_model.predict(X_val))

        score_val = accuracy
        scores_val_e.append(score_val)
        scores_train_e.append(scores.mean())
        e_l.append(e)
        md_l.append(mdv)

boost_dat = pd.DataFrame({'Validation_Metric':scores_val_e,'Train_Metric':scores_train_e,'Num_Estimator':e_l,'Max_Depth':md_l})
boost_dat_v = boost_dat[['Validation_Metric', 'Num_Estimator', 'Max_Depth']]
boost_dat_t = boost_dat[['Train_Metric', 'Num_Estimator', 'Max_Depth']]

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
fig.set_figheight(10)
fig.set_figwidth(10)
for label, df in boost_dat_t.groupby('Max_Depth'):
    df.plot('Num_Estimator','Train_Metric', ax=ax1, label=label)
ax1.set_title('AdaBoost Accuracy Performance Train - Product Subscription')
ax1.get_legend().remove()
fig.legend(title="Max Depth")
for label, df in boost_dat_v.groupby('Max_Depth'):
    df.plot('Num_Estimator','Validation_Metric', ax=ax2, label=label)
ax2.set_title('AdaBoost Accuracy Performance Validation - Product Subscription')
ax2.get_legend().remove()
#%%
#Learning Curve Adaboost
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,scoring = 'accuracy', n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Accuracy")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title('Learning Curve ANN - Product Subscription')
    ax.legend(loc="best")

    return plt

fig = plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
base = tree.DecisionTreeClassifier(max_depth= 2)
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
plt.title('Poly Kernel Type Optimal Degrees - Product Subscription')
plt.legend(loc='best')

# Compare Kernels
recall_score = []
recall_score_val = []
for k in kernels:
    if k == 'poly':
        svm_model = svm.SVC(kernel = 'poly', degree = 1)
    else:
        svm_model = svm.SVC(kernel = k)
    
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_val)
    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    scores = cross_val_score(svm_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    recall, precision, accuracy = calculate_precision_recall_accuracy(y_val.term_deposit_subscribed, y_pred)
    recall_score.append(scores.mean())
    recall_score_val.append(accuracy)

N = 3
ind = np.arange(N)
width = 0.35
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, recall_score, 0.35, color='royalblue', label = 'Training Data')
rects2 = ax.bar(ind + 0.35, recall_score_val, 0.35, color='seagreen', label = 'Validation Data')
plt.ylabel('Accuracy Score')
plt.title('SVM Kernel Types by Accuracy Score - Product Subscription')
plt.xticks(ind + width / 2, ('Poly (Linear)', 'RBF', 'Sigmoid'))
# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()


gamma_range = np.arange(0.01,0.3,0.02)
train_scores, valid_scores = validation_curve(
svm.SVC(kernel='poly'), X_train, y_train, param_name="gamma", param_range=gamma_range,
cv=5, scoring = 'accuracy')
train_scores_mean = np.mean(train_scores, axis=1)
validation_scores_mean = np.mean(valid_scores, axis=1)

plt.plot(gamma_range, train_scores_mean, label = 'Training Data')
plt.plot(gamma_range, validation_scores_mean, label = 'Validation Data')
plt.xlabel('Gamma')
plt.ylabel('Accuracy Score')
plt.title('Optimal Gamma SVM - Product Subscription')
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
    ax.title('Learning Curve SVM - Product Subscription')
    ax.legend(loc="best")

    return plt

fig = plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
model = svm.SVC(kernel = 'poly', degree = 1, gamma = 0.05)
plot_learning_curve(model, X_train, y_train, ax = fig, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

# %%
# Wall Clock times
# Decision Tree Model
dt_model = tree.DecisionTreeClassifier(min_samples_leaf = 55, max_depth = 5)
# KNN Model
knn_model = KNeighborsClassifier(n_neighbors= 18)
# AdaBoost Model
base = tree.DecisionTreeClassifier(max_depth= 2)
ada_model = AdaBoostClassifier(base_estimator=base, n_estimators = 20)
# ANN model
ann_model = MLPClassifier(hidden_layer_sizes=(3,128), activation = 'logistic', solver = 'adam', max_iter = 10)
# SVM Model
svm_model = svm.SVC(kernel = 'poly', degree = 1, gamma = 0.05)

models = {'Decision Tree' : dt_model, 'KNN' : knn_model, 'AdaBoost' : ada_model, 'ANN' : ann_model, 'SVM' : svm_model}


t = TicToc()
model_fit = []
model_query = []
test_recall = []
test_precision = []
test_accuracy = []
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

    recall, precision, accuracy = calculate_precision_recall_accuracy(y_test.term_deposit_subscribed, y_pred)
    test_recall.append(recall)
    test_precision.append(precision)
    test_accuracy.append(accuracy)
    model_l.append(k)

wc = pd.DataFrame({'model':model_l,
'fit_time':model_fit,
'query_time':model_query,
'accuracy': test_accuracy,
'recall':test_recall,
'precision':test_precision})

wc2 = wc.sort_values(by = ['fit_time'], ascending = False)
wc3 = wc.sort_values(by = ['accuracy'], ascending = False)

N = 5
ind = np.arange(N)
width = 0.35
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, wc2.fit_time, 0.35, label = 'Fit Time')
rects2 = ax.bar(ind + 0.35, wc2.query_time, 0.35, label = 'Query Time')
plt.ylabel('Seconds')
plt.title('Fit and Query Times by Model - Product Subscription')
plt.xticks(ind + width / 2, ('SVM', 'ANN', 'AdaBoost','Decision Tree','KNN'))
# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

N = 5
w = 0.25
ind = np.arange(N)
fig = plt.figure()
ax = fig.add_subplot(111)
rects = ax.bar(ind, wc3.accuracy, w, label = 'Accuracy')
rects1 = ax.bar(ind + w, wc3.recall, w, label = 'Recall')
rects2 = ax.bar(ind + (w*2), wc3.precision, w, label = 'Precision')
plt.ylabel('Scoring')
plt.title('Optimized Model Performances on Test Data - Product Subscription')
plt.xticks(ind + 0.35 / 2, ('Decision Tree', 'SVM', 'AdaBoost','KNN','ANN'))
# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()
# %%

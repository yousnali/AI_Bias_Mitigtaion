%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from datetime import date
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from fairml import audit_model
from fairml import plot_dependencies
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

# IBM's fairness toolbox:
from aif360.datasets import BinaryLabelDataset  # To handle the data
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric  # For calculating metrics
from aif360.explainers import MetricTextExplainer  # For explaining metrics
from aif360.algorithms.preprocessing import Reweighing  # Preprocessing technique

from IPython.display import Markdown, display
import seaborn as sns

sns.set_theme(style="ticks", color_codes=True)


path = '/Users/youssefennali/Desktop/Thesis/Python Projects/Stats/Final/'
filename = 'cox-violent-parsed_cleansed.csv'


# read the csv file
df = pd.read_csv(path+filename, sep=',', parse_dates = ['dob','c_jail_in','c_jail_out','c_offense_date','r_offense_date'])



df = pd.get_dummies(df, columns=['sex','race'], drop_first = False)

X = df[[
            'v_decile_score'
            # ,'decile_score' #removed since the target variable is based on this, otherwise the model will be overfitted
            ,'priors_count'
            ,'age'
            ,'sex_Male'
            # ,'sex_Female'
            # ,'race_African-American'
            # ,'race_Caucasian'
            # ,'race_Asian'
            # ,'race_Hispanic'
            # ,'race_Native American'
            # ,'race_Other'
            ,'african_american'
            # , 'african_american_0'
            # , 'african_american_1'
            ,'juv_misd_count'
            ,'juv_other_count'
            # ,'start'
            ,'is_recid'
            # , 'event' #no effect on the model's accuracy
            ,'days_in_jail' #this causes multicollinearity
        ]] 

#target variable
Y = df['medium_to_high_risk']

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.30)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)

# with sklearn
biasedReg = LogisticRegression(max_iter=700)
biasedReg.fit(x_train, y_train)

y_pred_biased = biasedReg.predict(x_test)

#evaluate model biased model

#accuracy assessment
print('Sex biased - Score (train set):', biasedReg.score(x_train,y_train))
print('Sex biased - Score (test set):', biasedReg.score(x_test,y_test))


scores = cross_val_score(biasedReg, X, Y, cv=5)
print('Sex biased - Cross-validated scores', scores)

accuracy = biasedReg.score(x_test, y_test)
print('Sex biased - Cross-predicted Accuracy:', accuracy)

#confusion matrix assessment biased model
cm_biased = metrics.confusion_matrix(y_test, y_pred_biased)

plt.figure(figsize=(9,9))
sns.heatmap(cm_biased, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Scores: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);


#XAI: feature importance 
importances, _ = audit_model(biasedReg.predict, X)

print(importances)

plot_dependencies(
    importances.median(),
    reverse_values=False,
    title="Biased - model feature dependence"
)


#Apply debiasing mitigation method
sex_privileged_groups = [{'sex_Male': 1}]
sex_unprivileged_groups = [{'sex_Male': 0}]

# Metric for the train dataset
train_biased = BinaryLabelDataset(df=pd.concat((x_train, y_train),
                                                axis=1),
                                  label_names=['medium_to_high_risk'],
                                  protected_attribute_names=['sex_Male'],
                                  favorable_label=1,
                                  unfavorable_label=0)

# Metric for the test dataset
test_biased = BinaryLabelDataset(df=pd.concat((x_test, y_test),
                                                axis=1),
                                  label_names=['medium_to_high_risk'],
                                  protected_attribute_names=['sex_Male'],
                                  favorable_label=1,
                                  unfavorable_label=0)



# Create the metric object for training set
metric_train_biased = BinaryLabelDatasetMetric(train_biased,
                                            unprivileged_groups=sex_unprivileged_groups,
                                            privileged_groups=sex_privileged_groups)

display(Markdown("#### Original training dataset"))
print("Sex biased - Difference in mean outcomes between unprivileged and privileged sex groups = %f" % metric_train_biased.mean_difference())

# Create the metric object for testing set
metric_test_biased = BinaryLabelDatasetMetric(test_biased,
                                            unprivileged_groups=sex_unprivileged_groups,
                                            privileged_groups=sex_privileged_groups)

display(Markdown("#### Original training dataset"))
print("Sex biased - Difference in mean outcomes between unprivileged and privileged sex groups = %f" % metric_test_biased.mean_difference())


#debias with the reweighing method
RW = Reweighing(unprivileged_groups=sex_unprivileged_groups,
                privileged_groups=sex_privileged_groups)
RW.fit(train_biased)
dataset_transf_train_f = RW.fit_transform(train_biased)

# Metric for the reweighted dataset
metric_reweigh = BinaryLabelDatasetMetric(dataset_transf_train_f, 
                                              unprivileged_groups=sex_unprivileged_groups,
                                              privileged_groups=sex_privileged_groups)
display(Markdown("#### Original training dataset"))
print("Sex debiased - Difference in mean outcomes between unprivileged and privileged sex groups = %f" % metric_reweigh.mean_difference())

debiasedReg = LogisticRegression(max_iter=700)
debiasedReg.fit(x_train, y_train, sample_weight= dataset_transf_train_f.instance_weights)

#evaluate debiased model

#accuracy assessment
print('Sex debiased - Score (train set):', debiasedReg.score(x_train,y_train))
print('Sex debiased - Score (test set):', debiasedReg.score(x_test,y_test))


scores_sex = cross_val_score(debiasedReg, X, Y, cv=5)
print('Sex debiased - Cross-validated scores', scores_sex)

accuracy_sex = debiasedReg.score(x_test, y_test)
print('Sex debiased - Cross-predicted Accuracy:', accuracy_sex)

#confusion matrix assessment debiased model
y_pred_debiased = debiasedReg.predict(x_test)

cm_debiased = metrics.confusion_matrix(y_test, y_pred_debiased)

plt.figure(figsize=(9,9))
sns.heatmap(cm_debiased, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Scores: {0}'.format(accuracy_sex)
plt.title(all_sample_title, size = 15);

importancesRW, _ = audit_model(debiasedReg.predict, x_train)

# print(importancesRW)

plot_dependencies(
    importancesRW.median(),
    reverse_values=False,
    title="Sex debiased - Model feature dependence"
)

print('Sex debiased - Score (train set):', debiasedReg.score(x_train,y_train))
print('Sex debiased - Score (test set):', debiasedReg.score(x_test,y_test))


#Reweigh race debias mitigation 
race_privileged_groups = [{'african_american': 0}]
race_unprivileged_groups = [{'african_american': 1}]

# Metric for the train dataset
train_race_biased = BinaryLabelDataset(df=pd.concat((x_train, y_train),
                                                axis=1),
                                  label_names=['medium_to_high_risk'],
                                  protected_attribute_names=['african_american'],
                                  favorable_label=0,
                                  unfavorable_label=1)

# Metric for the test dataset
test_race_biased = BinaryLabelDataset(df=pd.concat((x_test, y_test),
                                                axis=1),
                                  label_names=['medium_to_high_risk'],
                                  protected_attribute_names=['african_american'],
                                  favorable_label=0,
                                  unfavorable_label=1)



# Create the metric object for training set
metric_train_race_biased = BinaryLabelDatasetMetric(train_race_biased,
                                            unprivileged_groups=race_unprivileged_groups,
                                            privileged_groups=race_privileged_groups)

display(Markdown("#### Original training dataset"))
print("Race biased - Difference in mean outcomes between unprivileged and privileged race groups = %f" % metric_train_race_biased.mean_difference())

# debias with the reweighing method
RW_race = Reweighing(unprivileged_groups=race_unprivileged_groups,
                privileged_groups=race_privileged_groups)
RW_race.fit(train_race_biased)
dataset_transf_train_f_race = RW_race.fit_transform(train_race_biased)

# Metric for the reweighted dataset
metric_reweigh_race = BinaryLabelDatasetMetric(dataset_transf_train_f_race, 
                                              unprivileged_groups=race_unprivileged_groups,
                                              privileged_groups=race_privileged_groups)
display(Markdown("#### Original training dataset"))
print("Race debiased - Difference in mean outcomes between unprivileged and privileged race groups = %f" % metric_reweigh_race.mean_difference())

raceDebiasedReg = LogisticRegression(max_iter=700)
raceDebiasedReg.fit(x_train, y_train, sample_weight= dataset_transf_train_f_race.instance_weights)

#evaluate debiased model

#confusion matrix assessment debiased model
y_pred_race_debiased = raceDebiasedReg.predict(x_test)

cm_race_debiased = metrics.confusion_matrix(y_test, y_pred_race_debiased)

plt.figure(figsize=(9,9))
sns.heatmap(cm_debiased, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Scores: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);

importances_race_RW, _ = audit_model(raceDebiasedReg.predict, x_train)

# print(importancesRW)

plot_dependencies(
    importances_race_RW.median(),
    reverse_values=False,
    title="Race debiased - Model feature dependence"
)


#accuracy assessment
print('Race debiased - Score (train set):', raceDebiasedReg.score(x_train,y_train))
print('Race debiased - Score (test set):', raceDebiasedReg.score(x_test,y_test))


scores_race = cross_val_score(raceDebiasedReg, X, Y, cv=5)
print('Race debiased - Cross-validated scores', scores_race)

accuracy_race = raceDebiasedReg.score(x_test, y_test)
print('Race debiased - Cross-predicted Accuracy:', accuracy_race)








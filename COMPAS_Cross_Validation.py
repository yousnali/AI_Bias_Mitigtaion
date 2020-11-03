%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from datetime import date
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from fairml import audit_model
from fairml import plot_dependencies
import seaborn as sns

path = '/Users/youssefennali/Desktop/Thesis/Python Projects/Stats/Final/'
filename = 'cox-violent-parsed_cleansed.csv'


# read the csv file
df = pd.read_csv(path+filename, sep=',', parse_dates = ['dob','c_jail_in','c_jail_out','c_offense_date','r_offense_date'])



df = pd.get_dummies(df, columns=['sex','race'], drop_first = True)

X = df[[
            'v_decile_score'
            # ,'decile_score' #removed since the target variable is based on this, otherwise the model will be overfitted
            ,'priors_count'
            ,'age'
            # ,'sex_Male'
            # ,'sex_Female'
            # ,'race_African-American'
            # ,'race_Caucasian'
            # ,'race_Asian'
            # ,'race_Hispanic'
            # ,'race_Native American'
            # ,'race_Other'
            ,'juv_misd_count'
            ,'juv_other_count'
            # ,'event' #no effect on the model's accuracy
            ,'is_recid'
            ,'days_in_jail' 
        ]] 

#target variable
Y = df['medium_to_high_risk']

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.30)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)

# with sklearn
logisticRegr = LogisticRegression(max_iter=700)
logisticRegr.fit(x_train, y_train)

y_pred = logisticRegr.predict(x_test)

#evaluate model
print('Score (train set):', logisticRegr.score(x_train,y_train))
print('Score (test set):', logisticRegr.score(x_test,y_test))


scores = cross_val_score(logisticRegr, X, Y, cv=5)
print('Cross-validated scores', scores)

accuracy = logisticRegr.score(x_test, y_test)
print('Cross-predicted Accuracy:', accuracy)

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Scores: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);


importances, _ = audit_model(logisticRegr.predict, X)

print(importances)

plot_dependencies(
    importances.median(),
    reverse_values=False,
    title="Model feature dependence"
)


CorrMatrix = df.corr(method="pearson", min_periods=1)
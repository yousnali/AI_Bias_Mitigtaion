%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(style="ticks", color_codes=True)


path = '/Users/youssefennali/Desktop/Thesis/Python Projects/Data Sets/'
filename = 'cox-violent-parsed.csv'


# read the csv file
df = pd.read_csv(path+filename, sep=',', parse_dates = ['dob','c_jail_in','c_jail_out','c_offense_date','r_offense_date'])

#drop negative is recid value
df = df[df['is_recid'] >= 0]
df = df[df['decile_score'] >= 0]
df = df[df['v_decile_score'] >= 0]
df = df[df['priors_count'] >= 0]

#transform to dummies
# df = pd.get_dummies(df, columns=['sex','race'], drop_first=False)

#drop empty column violent_recid and duplicate columns
# del df['is_violent_recid_1']
del df['priors_count.1']
del df['decile_score.1']
del df['violent_recid']
del df['id']
# del df['event'] #remove because it's strongly related to is_violent_recid variable
# del df['start'] #remove because it's strongly related to is_violent_recid variable
# del df['end'] #remove because it's strongly related to is_violent_recid variable

# Remove rows with empty jail dates
df.dropna(subset=['c_jail_in'], inplace=True)

# Remove rows with an in jail in date after an out jail date
df = df.drop(df[df.c_jail_out < df.c_jail_in].index)

# Remove timestamps from date
df['c_jail_out'] = df['c_jail_out'].dt.date
df['c_jail_in'] = df['c_jail_in'].dt.date

# Number days in jail transformation
df['days_in_jail'] = df['c_jail_out'] - df['c_jail_in'] 
df['days_in_jail'] = pd.to_numeric((df['days_in_jail'] / np.timedelta64(1, 'D')),downcast='integer')

#transform target variable

# create a list of conditions for medium_to_high_risk
conditions = [
    (df['score_text'] == 'Low'),
    (df['score_text'] == 'Medium'),
    (df['score_text'] == 'High')
    ]

# create a list of the values we want to assign for each condition
values = [0, 1, 1]

# create a new column and use np.select to assign values to it using our lists as arguments
df['medium_to_high_risk'] = np.select(conditions, values)

# create a list of conditions for medium_to_high_risk
race_conditions = [
    (df['race'] == 'African-American'),
    (df['race'] == 'Caucasian'),
    (df['race'] == 'Asian'),
    (df['race'] == 'Hispanic'),
    (df['race'] == 'Native American'),
    (df['race'] == 'Other')
    ]

# create a list of the values we want to assign for each condition
race_values = [1, 0, 0, 0, 0, 0]

# create a new column and use np.select to assign values to it using our lists as arguments
df['african_american'] = np.select(race_conditions, race_values)

df.to_csv('/Users/youssefennali/Desktop/Thesis/Python Projects/Stats/Final/cox-violent-parsed_cleansed.csv')


CorrMatrix = df.corr(method="pearson", min_periods=1)

CorrMatrix.to_csv('/Users/youssefennali/Desktop/Thesis/Python Projects/Data Sets/temp/compas_correlation.csv')

# sns.heatmap(df.iloc[:,:50].corr())
# sns.heatmap(df.iloc[:,50:].corr())




# print(CorrMatrix)

ax = sns.heatmap(
    CorrMatrix, xticklabels=True, yticklabels=True,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
);

# iterating the columns 
# for col in CorrMatrix.columns: 
#     print(col) 
    


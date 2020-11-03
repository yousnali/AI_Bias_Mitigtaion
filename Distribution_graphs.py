import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import statistics as stat

import seaborn as sns

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
del df['event'] #remove because it's strongly related to is_violent_recid variable
del df['start'] #remove because it's strongly related to is_violent_recid variable
del df['end'] #remove because it's strongly related to is_violent_recid variable
del df['is_violent_recid'] #remove because it's strongly related to is_violent_recid variable
del df['juv_other_count'] #remove because it's strongly related to is_violent_recid variable
del df['juv_misd_count'] #remove because it's strongly related to is_violent_recid variable
del df['juv_fel_count'] #remove because it's strongly related to is_violent_recid variable
del df['c_days_from_compas'] #remove because it's strongly related to is_violent_recid variable
del df['days_b_screening_arrest'] #remove because it's strongly related to is_violent_recid variable


# Remove rows with empty jail dates
df.dropna(subset=['c_jail_in'], inplace=True)

# Remove rows with an in jail in date after an out jail date
df = df.drop(df[df.c_jail_out < df.c_jail_in].index)

# Remove timestamps from date
df['c_jail_out'] = df['c_jail_out'].dt.date
df['c_jail_in'] = df['c_jail_in'].dt.date

# Number days in jail transformation
# df['days_in_jail'] = df['c_jail_out'] - df['c_jail_in'] 
# df['days_in_jail'] = pd.to_numeric((df['days_in_jail'] / np.timedelta64(1, 'D')),downcast='integer')
# del df['days_in_jail'] #remove because it's strongly related to is_violent_recid variable

# sns.displot(df, x="v_decile_score", discrete=True)
# sns.set_style("ticks")
# plt.title("Violence score distribution")
# plt.ylabel("frequency")
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# sns.displot(df, x="decile_score", discrete=True)
# sns.set_style("ticks")
# plt.title("Redivism score distribution")
# plt.ylabel("frequency")
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# sns.displot(df, x="priors_count", discrete=True)
# sns.set_style("ticks")
# plt.title("Priors distribution")
# plt.ylabel("frequency")
# plt.xlim(0,30)
# plt.xticks([0, 5, 10, 15, 20, 25, 30])

# sns.displot(df, x="age", discrete=True)
# sns.set_style("ticks")
# plt.title("Age distribution")
# plt.ylabel("frequency")
# plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])


print(df)

# # means, median, mode
# print(round(stat.mean(df['decile_score']), 2))
# print(round(stat.median(df['decile_score']),2))
# print(round(stat.mode(df['decile_score']),2))

# print(round(stat.mean(df['v_decile_score']),2))
# print(round(stat.median(df['v_decile_score']),2))
# print(round(stat.mode(df['v_decile_score']),2))

# # print(round(stat.mean(df['age']),2))
# # print(round(stat.median(df['age']),2))
# # print(round(stat.mode(df['age']),2))

# print(round(stat.mean(df['priors_count']),2))
# print(round(stat.median(df['priors_count']),2))
# print(round(stat.mode(df['priors_count']),2))




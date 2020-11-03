import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

path = '/Users/youssefennali/Desktop/Thesis/Python Projects/Stats/Final/'
filename = 'cox-violent-parsed_cleansed.csv'


# read the csv file
df = pd.read_csv(path+filename, sep=',', parse_dates = ['dob','c_jail_in','c_jail_out','c_offense_date','r_offense_date'])



df = pd.get_dummies(df, columns=['sex','race'], drop_first = False)
                                 # ,'is_recid','is_violent_recid'], drop_first=False)

X = df[[
            'v_decile_score'
            ,'decile_score'
            ,'priors_count'
            ,'age'
            # ,'is_recid_1'
            ,'sex_Male'
            ,'sex_Female'
            ,'race_African-American'
            ,'race_Caucasian'
            ,'race_Asian'
            ,'race_Hispanic'
            ,'race_Native American'
            ,'race_Other'
            ,'juv_misd_count'
            ,'juv_other_count'
            ,'event'
            ,'is_recid'
            ,'days_in_jail' #this causes multicollinearity
            ,'decile_score'
            ,'medium_to_high_risk'
            ,'african_american'
        ]] 

#target variable
Y = df['medium_to_high_risk']

#Here we can see that X1 and X2 have a high and similar correlation coefficient
#(Also X3 and X4 have similar coefficients but they are lower so we can allow low collinearity)


#Method 2 to Detect MultiCollinearity

def get_VIF(X , target):
    X = add_constant(X.loc[:, X.columns != 'medium_to_high_risk'])
    seriesObject = pd.Series([variance_inflation_factor(X.values,i) for i in range(X.shape[1])] , index=X.columns,)
    return seriesObject

target = Y
print(get_VIF(X,target))

#Here we Observe that X1 and X2 are having VIF value of infinity so we need to drop one of them
#(Any value greater than 5-6 shows MultiCollinearity)
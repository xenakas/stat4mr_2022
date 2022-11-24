"""
Title:  Data Generating Processes
Author: Matteo Courthoud
Date:   24/03/2022
"""


import numpy as np
import scipy as sp
import pandas as pd

from numpy.random import normal as rnd
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler

class dgp_ad():
    """
    Data Generating Process: ads
    """
    
    def __init__(self):
        self.Y = 'revenue'
        self.T = 'ad_exposure'
        self.X = ['male', 'black', 'age', 'educ']
    
    def generate_data(self, seed=1, N=1000, oracle=False):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame({'male': np.random.binomial(1, 0.5, N),
                           'black': np.random.binomial(1, 0.5, N),
                           'age': np.rint(np.random.normal(45, 10, N))})

        # Endogenous observables
        df['educ'] = np.random.poisson(2*df['black'] + 1)

        # Treatment
        df[self.T] = (np.random.uniform(0, 2, N) \
                    + 0.3 * df['male'] \
                    - 0.2 * df['black']) > 1

        # Treatment effect
        Y0 =  0.5 * df['male'] \
            - 0.5 * df['black'] \
            + 0.1 * np.log(1 + df['educ']) \
            - 0.2 * (df['age']>50) \
            + rnd(0, 1, N)
        Y1 = Y0 \
            + 0.5 \
            + 0.4 * df['male'] \
            + 0.1 * np.log(1 + df['educ']) \
            + 0.2 * (df['age']>40)
        if oracle:
            df['Y0'] = Y0
            df['Y1'] = Y1
        df[self.Y] = Y0 + Y1 * df[self.T]
       
        return df


class dgp_aipw():
    """
    Data Generating Process for AIPW
    
    from https://www.youtube.com/watch?v=IfZHUFFlsGc
    """
    
    def __init__(self, p=20):
        self.p = p
        self.Y = 'Y'
        self.T = 'T'
        self.X = [f"x{i}" for i in range(1,p+1)]
    
    def generate_data(self, seed=1, N=1000):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame(np.random.normal(0, 1, (N, self.p)), columns=self.X)

        # Propensity score
        df['e'] = 1 / (1 + np.exp(- df['x1']))

        # Treatment
        df['T'] = (np.random.uniform(0, 1, N) < df['e']).astype(int)

        # Outcomes
        df['Y0'] = np.maximum(df['x1'] + df['x2'], 0) - 0.05 * df['T']
        df['Y1'] = np.maximum(df['x1'] + df['x3'], 0) - 0.05 * df['T'] - 0.05
        df['Y'] = df['Y0'] * (1-df['T']) + df['Y1'] * df['T']

        return df


class dgp3():
    """
    Data Generating Process 3
    """
    
    def __init__(self, p=20):
        self.p = p
        self.Y = 'Y'
        self.T = 'T'
        self.X = [f"x{i}" for i in range(1,p+1)]
    
    def generate_data(self, seed=1, N=1000):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame(np.random.normal(0, 1, (N, self.p)), columns=self.X)

        # Propensity score
        df['e'] = 1 / (1 + np.exp(- df['x1']))

        # Treatment
        df['T'] = (np.random.uniform(0, 1, N) < df['e']).astype(int)

        # Outcomes
        df['Y'] = np.maximum(df['x1'] + df['x2'] * (1-df['T']) + df['x3'] * df['T'], 0) - 0.05 * df['T']

        return df


class dgp4():
    """
    Data Generating Process 4
    
    from https://www.youtube.com/watch?v=N9ThAs7NS0g
    """
    
    def __init__(self, p=10):
        self.p = p
        self.Y = 'Y'
        self.T = 'T'
        self.X = [f"x{i}" for i in range(1,p+1)]
    
    def generate_data(self, seed=1, N=4000):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame(np.random.normal(0, 1, (N, self.p)), columns=self.X)

        # Treatment probability
        df['e'] = 0.3

        # Treatment assignment
        df['T'] = (np.random.uniform(0, 1, N) < df['e']).astype(int)
        
        # Treatment effect
        df['tau'] = 1 / (1 + np.exp(-df['x3']))

        # Outcomes
        df['Y'] = np.maximum(df['x1'] + df['x2'], 0) + df['T'] * df['tau'] + np.random.normal(0, 1, N)

        return df


class dgp_newsletter():
    """
    Data Generating Process: reminder to sign up to newsletter
    """
    
    def __init__(self):
        self.Z = 'reminder'
        self.T = 'subscribe'
        self.Y = 'revenue'
        self.X = []
    
    def generate_data(self, seed=1, N=1000, oracle=False):
        np.random.seed(seed)
        
        # Nudge / instrument
        df = pd.DataFrame({self.Z: np.random.binomial(1, 0.5, N)})
        
        # Hidden type
        income = np.random.exponential(1, N)
        if oracle: df['income'] = income
            
        # Treatment assignment
        df[self.T] = (expit(- income + df[self.Z] + rnd(size=N)) > 0.5).astype(int)
        
        # Treatment effect
        tau = 1
        if oracle: df['tau'] = tau

        # Outcome
        df[self.Y] = -1 + 2*income + tau*df[self.T] + rnd(size=N)

        return df



class dgp_membership():
    """
    Data Generating Process: Incentives for membership
    
    inspired by: https://github.com/microsoft/EconML/blob/main/notebooks/CustomerScenarios
    """
    
    def __init__(self):
        self.Z = 'easier_signup'
        self.T = 'became_member'
        self.Y = 'revenue_post'
        self.X = []
    
    def generate_data(self, seed=1, N=100000, oracle=False):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame({'visit_flights': np.random.randint(0, 28, N),
                           'visit_hotels': np.random.randint(0, 28, N),
                           'visit_restaurants': np.random.randint(0, 28, N),
                           'visit_rental': np.random.randint(0, 28, N),
                           'origin_US': np.random.binomial(1, 0.7, N),
                           'mobile': np.random.binomial(1, 0.3, N),
                           'revenue_pre': np.random.exponential(1, N)})
        self.X = df.columns
        
        # Hidden type
        income = np.random.exponential(1, N)

        # Nudge / instrument
        df[self.Z] = np.random.binomial(1, 0.5, N)

        # Treatment probability
        e = expit(- income + df[self.Z] + rnd(size=N))
        if oracle: df['e'] = e
        
        # Treatment assignment
        df[self.T] = (np.random.uniform(0, 1, N) < e).astype(int)
        
        # Treatment effect
        tau = 0.2 + 0.3 * df['visit_flights'] - 0.2*df['visit_rental'] + df['mobile']
        if oracle: df['tau'] = tau

        # Outcome
        df[self.Y] = 1 + df['revenue_pre'] + 2*income + tau*df[self.T] + rnd(size=N)

        return df.round(2)
    

class dgp_ao18():
    """
    Data Generating Process: Anna and Olken (2018)
    
    inspired by: https://github.com/microsoft/EconML/blob/main/notebooks/CustomerScenarios
    """
    
    def __init__(self):
        self.T = 'cash_transfer'
        self.Y = 'welfare'

    def import_data(self, seed=1, oracle=False):
        np.random.seed(seed)

        # Import data
        df = pd.read_csv('data/ao18.csv')
        
        self.X = [c for c in df.columns if c not in ['hhid', 'consumption_0']]

        # Treatment 
        df['cash_transfer'] = np.random.binomial(1, 0.5, len(df))

        # Treatment effect
        df['consumption'] = df['consumption_0'] + df['cash_transfer']*100

        # Hide 
        df['welfare'] = np.log(df['consumption'])

        # Hide variables
        if oracle:
            df['welfare_0'] = np.log(df['consumption_0'])
            df['welfare_1'] = np.log(df['consumption_0'] + 100)
        else:
            df = df.drop(columns=['consumption_0'])

        # Return
        return df.dropna()
    

class dgp_did():
    """
    Data Generating Process: TV advertising
    
    Inspired by: https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html
    """
    
    def __init__(self):
        self.D = 'treated'
        self.Y = 'revenue'
        self.T = 'days'

    def generate_data(self, seed=1, N=100, T=20, oracle=False):
        np.random.seed(seed)

        # Init data
        df = pd.DataFrame(np.array(np.meshgrid(range(1, T+1), range(1,N), [0,1])).T.reshape(-1,3), 
                  columns=['day', 'id', 'treated'])
        #df['treated'] = df['treated'].astype('bool')
        
        
        # Treatment
        alpha_i = np.sqrt(df['id']) - 3*(df['id']>10)
        gamma_t = 0.1*df['day'] + rnd(size=len(df))
        tau_it = 0.5*np.log(1+df['id']) - 0.12*df['day']
        
        # Effect
        df['post'] = (df['day'] > T/2)
        df['revenue'] = alpha_i + gamma_t + \
                        1.2*df['treated'] + \
                        df['post']*df['treated']*tau_it + \
                        rnd(size=len(df))
        
        # Hide variables
        if oracle:
            df['alpha_i'] = alpha_i
            df['gamma_t'] = gamma_t
            df['tau_it'] = tau_it

        # Return
        return df
    
    
class dgp_school():
    """
    Data Generating Process: class size and test scores
    """
    
    def __init__(self):
        self.T = 'class_size'
        self.Y = 'math_score'
    
    def generate_data(self, seed=1, N=1000, oracle=False):
        np.random.seed(seed)
        
        # Dataframe
        df = pd.DataFrame({'math_hours': np.random.randint(2,5,N),
                           'history_hours': np.random.randint(2,5,N),
                           'good_school': np.random.binomial(1,0.5,N),
                           'class_year': np.random.randint(1,5,N)})
        
        # Treatment
        df[self.T] = np.random.poisson(25, N) - 1*df['class_year'] - 7*df['good_school']
        
        # Hidden ability
        ability = np.random.exponential(1, N)
        history_hours = np.random.randint(3,5,N)
            
        # Main outcome
        df[self.Y] = 1 + 0.2*df[self.T] + ability + df['math_hours'] + 5*df['good_school'] + rnd(size=N)
        
        # Other outcome
        df['hist_score'] = 1 + 0.2*df[self.T] + ability + history_hours + 5*df['good_school'] + rnd(size=N)
                
        return df
    

class dgp_marketplace():
    """
    Data Generating Process: online marketplace
    """
    
    def generate_data(self, seed=1, N=10_000):
        np.random.seed(seed)
        
        # Does the firm sells only online?
        online = np.random.binomial(1, 0.5, N)
        
        # How many products does the firm have
        products = 1 + np.random.poisson(1, N)
        
        # What is the age of the firm
        t = np.random.exponential(0.5*products, N) 
        
        # Sales
        sales = 1e3 * np.random.exponential(products + np.maximum((1 + 0.3*products + 4*online)*t - 0.5*(1 + 6*online)*t**2, 0), N)

        # Generate the dataframe
        df = pd.DataFrame({'age': t, 'sales': sales, 'online': online, 'products': products})
                
        return df
    
    
    
class dgp_store_coupons():
    """
    Data Generating Process: store coupons
    """
    
    def generate_data(self, seed=1, N=300, K=5):
        np.random.seed(seed)
        
        # Incomme
        income = np.round(np.random.normal(50, 10, N), 3) 
        
        # Using a coupon
        coupons = np.round(np.random.normal(0.5, 0.1, N) - income / 200, 3)
        
        # Day of the week
        day = np.random.choice(range(1,8), N)
        
        # Sales
        sales = np.round(10 * (income + 20*coupons + day + np.random.normal(10, 2, N)), 1)

        # Generate the dataframe
        df = pd.DataFrame({'sales': sales, 'coupons': coupons, 'income': income, 'dayofweek': [str(d) for d in day]})

        return df

    
class dgp_educ_wages():
    """
    Data Generating Process: aducation and wages
    """
    
    def generate_data(self, seed=1, N=300):
        np.random.seed(seed)
        
        # Ability
        ability = np.round(np.random.uniform(0, 10, N), 3) 
        
        # Controls 
        age = np.random.randint(25, 65, N)
        gender = np.random.choice(['male', 'female'], N)
        
        # Education
        education = np.random.randint(5, 10, N) + ability//3
        
        # Wage
        wage = 100 * np.round(ability/2 + education + 8*np.log(age) + 2*(gender=='male') + np.random.normal(0, 4, N))
        
        # Generate the dataframe
        df = pd.DataFrame({'age': age, 'gender': gender, 'education': education, 'wage': wage})

        return df
    
    
class dgp_pretest():
    """
    Data Generating Process: pre-test bias
    """
    
    def generate_data(self, a=1, b=.3, c=3, N=1000, seed=1):
        np.random.seed(seed)
        
        # Past Sales
        past_sales = np.random.normal(5, 1, N)
        
        # Advertisement 
        ads = c*past_sales + np.random.normal(-3, 1, N)
        
        # Education
        sales = a*ads + b*past_sales + np.random.normal(0, 1, N)
                
        # Generate the dataframe
        df = pd.DataFrame({'ads': ads, 'sales': sales, 'past_sales': past_sales})

        return df
    
    
class dgp_rnd_assignment():
    """
    Data Generating Process: random assignment 
    """
    
    def generate_data(self, N=1000, seed=1):
        np.random.seed(seed)
        
        # Treatment assignment
        group = np.random.choice(['treatment', 'control'], N, p=[0.3, 0.7])
        arm_number = np.random.choice([1,2,3,4], N)
        arm = [f'arm {n}' for n in arm_number]

        # Covariates 
        gender = np.random.binomial(1, 0.5 + 0.1*(group=='treatment'), N) 
        age = np.rint(18 + np.random.beta(2 + (group=='treatment'), 5, N)*50)
        mean_income = 6 + 0.1*arm_number
        var_income = 0.2 + 0.1*(group=='treatment')
        income = np.round(np.random.lognormal(mean_income, var_income, N), 2)

        # Generate the dataframe
        df = pd.DataFrame({'Group': group, 'Arm': arm, 'Gender': gender, 'Age': age, 'Income': income})
        df.loc[df['Group']=='control', 'Arm'] = np.nan

        return df


class dgp_buttons():
    """
    Data Generating Process: buttons
    """
    
    def __init__(self):
        self.effects = [1,-4]
        self.groups = [' default', 'button1', 'button2']
    
    def generate_data(self, N=1000, seed=1, truth=False):
        np.random.seed(seed)
        
        # Device group
        mobile = np.random.binomial(1, 0.5, N)
        
        # Treatment assignment
        group = pd.Series(mobile)
        group[mobile==True] = np.random.choice(self.groups, p=[0.4, 0.2, 0.4], size=sum(mobile==True))
        group[mobile==False] = np.random.choice(self.groups, p=[0.4, 0.4, 0.2], size=sum(mobile==False))
        
        # Effects
        effect1 = np.random.normal(self.effects[0]*(mobile==True), 1)
        effect2 = np.random.normal(self.effects[1]*(mobile==False), 1)
        revenue = (effect1 + effect2)*(group==self.groups[2]) + 3*mobile + np.random.normal(10, 1, N)
                
        # Generate the dataframe
        df = pd.DataFrame({'group': group, 'revenue': revenue, 'mobile': mobile})
        
        # Add true effects
        if truth: 
            df['effect'] = (effect1 + effect2)*(group=='treat2')

        return df
    
    
class dgp_cuped():
    """
    Data Generating Process: CUPED
    """
    
    def __init__(self, alpha=5, beta=0, gamma=3, delta=2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
    
    def generate_data(self, N=100, seed=1):
        np.random.seed(seed)
        
        # Individuals
        i = range(1,N+1)

        # Treatment status
        d = np.random.binomial(1, 0.5, N)
        
        # Individual outcome pre-treatment
        y0 = self.alpha + self.beta*d + np.random.normal(0, 1, N)
        y1 = y0 + self.gamma + self.delta*d + np.random.normal(0, 1, N)

        # Generate the dataframe
        df = pd.DataFrame({'i': i, 'ad_campaign': d, 'revenue0': y0, 'revenue1': y1})

        return df
    

class dgp_darkmode():
    """
    Data Generating Process: blog dark mode and time spend reading
    """
    
    def generate_data(self, N=300, seed=1):
        np.random.seed(seed)
        
        # Control variables
        male = np.random.binomial(1, 0.45, N)
        age = np.rint(18 + np.random.beta(2, 2, N)*50)
        hours = np.minimum(np.round(np.random.lognormal(5, 1.3, N), 1), 2000)
        
        # Treatment
        pr = np.maximum(0, np.minimum(1, 0.8 + 0.3*male - np.sqrt(age-18)/10))
        dark_mode = np.random.binomial(1, pr, N)==1
        
        # Outcome
        read_time = np.round(np.random.normal(10 - 4*male + 2*np.log(hours) + 2*dark_mode, 4, N), 1)

        # Generate the dataframe
        df = pd.DataFrame({'read_time': read_time, 'dark_mode': dark_mode, 'male': male, 'age': age, 'hours': hours})

        return df
    
    
class dgp_compare():
    """
    Data Generating Process: a lot of stuff together
    """
    
    def generate_data(self, N=10000, seed=1, include_beta=False):
        np.random.seed(seed)
        
        # Control variables
        male = np.random.binomial(1, 0.5, N)
        age = np.rint(18 + np.random.beta(2, 2, N)*50)
        income = np.rint(np.random.lognormal(7.5, .3, N))
        
        # Treatment assignment
        pr = np.maximum(0, np.minimum(1, 0.55 - 0.1*male + np.sqrt(age)/3 - np.log(income)/3.6))
        d = np.random.binomial(1, pr, N)==1
        
        # Treatment effect
        beta = np.random.normal(3*male - np.sqrt(age) + 2*np.log(income))
        beta = beta - np.mean(beta) + 2
        
        # Outcome
        y = np.round(np.random.normal(20 + 3*male - np.sqrt(age) + 2*np.log(income) + beta*d, 5, N), 2)

        # Generate the dataframe
        df = pd.DataFrame({'outcome': y, 'treated': d, 'male': male, 'age': age, 'income': income})
        if include_beta:
            df['beta'] = beta

        return df


class dgp_premium():
    """
    Data Generating Process: premium
    """
    
    def generate_data(self, N=300, seed=1, true_te=False):
        np.random.seed(seed)
        
        # Control variables
        age = np.round(np.random.uniform(18, 60, N), 2)
        
        # Treatment
        premium = np.random.binomial(1, 0.1, N)==1
        
        # Heterogeneous effects
        y0 = 10 + 0.1*(30<age)*(age<50)
        y1 = 0.5 + 0.3*(35<age)*(age<45)
        
        # Outcome
        revenue = np.round(np.random.normal(y0 + premium*y1, 0.15, N), 2)

        # Generate the dataframe
        df = pd.DataFrame({'revenue': revenue, 'premium': premium, 'age': age})
        
        # Add truth
        if true_te:
            df['y0'] = y0
            df['y1'] = y1

        return df
    
    
    
class dgp_selfdriving():
    """
    Data generating process: self-driving cars
    """
    
    def clean_data():
        df = pd.read_csv('../data/us_cities_20022019.csv')
        df = df[['Metropolitan areas', 'Variables', 'Year', 'Value']]
        df.columns = ['city', 'variable', 'year', 'value']
        df.loc[df['variable'].str.contains('Employment'), "variable"] = "employment"
        df.loc[df['variable'].str.contains('Population density'), "variable"] = "density"
        df.loc[df['variable'].str.contains('Population'), "variable"] = "population"
        df.loc[df['variable'].str.contains('GDP'), "variable"] = "gdp"
        df = pd.pivot(data=df, index=['city', 'year'], columns='variable').reset_index()
        df.columns = [''.join(col).replace('value', '') for col in df.columns]
        df['employment'] = df['employment'] / df['population']
        df['city'] = df['city'].str.replace('\(\w+\)', '').str.strip()
        df['population'] = df['population'] / 1e6
        df['gdp'] = df['gdp'] / 1e4
        df.to_csv('../data/us_cities_20022019_clean.csv', index=False)
        
        
    def generate_data(self, city='Chicago', year=2010, seed=1):
        np.random.seed(seed)
        
        # Load Data
        df = pd.read_csv('../data/us_cities_20022019_clean.csv')
        df = df[df['year']>2002]

        # Select only big cities
        df['mean_pop'] = df.groupby('city')['population'].transform('mean')
        df = df[df['mean_pop'] > 1].reset_index(drop=True)
        del df['mean_pop']

        # Treatment
        df['treated'] = df['city']==city
        df['post'] = df['year']>=year

        # Generate revenue
        df['revenue'] = df['gdp'] + np.sqrt(df['population']) + \
            20*np.sqrt(df['employment']) - df['density']/100 + \
            (df['year']-1990)/5 + np.random.normal(0,1,len(df)) + \
            df['treated'] * df['post'] * np.log(np.maximum(2, df['year']-year))
        
        return df


class dgp_p2p():
    """
    Data Generating Process: blog dark mode and time spend reading
    """

    def generate_data(self, N=50, seed=2):
        np.random.seed(seed)

        # Hours spent in game
        hours = 2 + np.round(np.random.normal(1, 1, N), 1)

        # Transactions
        transactions = np.round(np.random.normal(3*hours, 0.5, N), 2)

        # Generate the dataframe
        df = pd.DataFrame({'hours': hours, 'transactions': transactions})

        # Generate outliers
        df.loc[1,:] = [2, 8]
        df.loc[2,:] = [7, 21]
        df.loc[3,:] = [6.7, 18]

        return df
    
    
class dgp_credit():
    """
    Data Generating Process: credit cards
    """

    def generate_data(self, N=100, seed=0):
        np.random.seed(seed)
        
        # Connection speed
        connection = np.random.lognormal(3, 1, N)
        
        # Treatment assignment
        newUI = np.random.binomial(1, 0.5, N)
        
        # Transfer speed
        transfer = np.minimum(np.random.exponential(10 + 4*newUI - 0.5*np.sqrt(connection), N), connection)
        transfer = np.minimum(np.random.lognormal(2.8 + newUI, 1, N), connection)
        
        # Generate the dataframe
        df = pd.DataFrame({'newUI': newUI,  
                           'connection': np.round(connection,2), 
                           'transfer': np.round(transfer,2)})

        return df


class dgp_loyalty():
    """
    Data Generating Process: loyalty card
    """

    def generate_data(self, seed=1, N=10_000):
        np.random.seed(seed)

        # Treatment
        age = np.random.randint(18, 55, N)
        gender = np.random.choice(['Male', 'Female'], p=[0.6, 0.4], size=N)
        income = np.random.lognormal(4 + np.log(age), 0.1, N)
        loyalty = np.random.binomial(1, 0.5, N)

        # Spend
        spend = 50*(gender=='Female') + income/10 + loyalty*np.sqrt(age)
        spend = np.maximum(np.round(spend, 2) - 220, 0)

        # Generate the dataframe
        df = pd.DataFrame({'loyalty': loyalty, 'spend': spend, 'age': age, 'gender': gender})

        return df
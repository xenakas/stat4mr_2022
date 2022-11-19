import seaborn as sns
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.stats import norm
import scipy.stats as scs
from scipy.stats import binom, bernoulli

import pandas as pd
import numpy as np

    
def calculate_SE(p_A, p_B, n_A, n_B):
    p_pool = (p_A * n_A + p_B * n_B) / (n_A + n_B)
    #SE = np.sqrt(p_A * (1-p_A) / n_A + p_B * (1-p_B) / n_B)
    SE = np.sqrt(p_pool * (1-p_pool) * (1/n_A + 1/n_B))
    return SE


def generate_data(size, split, bcr, lift):
    df = pd.DataFrame(bernoulli.rvs(0.5, size=2000), columns=['group']).replace({0:'A', 1: 'B'})
    df.loc[df['group'] == 'A', 'clicked'] = bernoulli.rvs(bcr, size=len(df[df['group'] == 'A']))
    df.loc[df['group'] == 'B', 'clicked'] = bernoulli.rvs(bcr+lift, size=len(df[df['group'] == 'B']))
    
    summary = df.pivot_table(index='group', values='clicked', aggfunc=np.sum)
    summary['count'] = df.pivot_table(index='group', values='clicked', aggfunc=lambda x: len(x))
    summary['ctr'] = summary['clicked'] / summary['count']
    return df, summary

def generate_data_equal_size(size, bcr, lift, summary_sh=True):

    df = pd.DataFrame(['A'] * int(size/2) + ['B'] * int(size/2), columns=['group'])
    
    df.loc[df['group'] == 'A', 'clicked'] = bernoulli.rvs(bcr, size=len(df[df['group'] == 'A']))
    df.loc[df['group'] == 'B', 'clicked'] = bernoulli.rvs(bcr+lift, size=len(df[df['group'] == 'B']))
    
    summary = df.pivot_table(index='group', values='clicked', aggfunc=np.sum)
    summary['count'] = df.pivot_table(index='group', values='clicked', aggfunc=lambda x: len(x))
    summary['ctr'] = summary['clicked'] / summary['count']
    if summary_sh==True:
        print(summary)
    return df, summary

def plot_double_binomial(n_A, p_A, n_B, p_B):
    upper = int(n_B*p_B + 50)
    lower = int(n_A*p_A - 50)
    xx = np.arange(lower, upper)
    
    fig, ax = plt.subplots(figsize=(8,5))
    data_a = binom(n_A, p_A).pmf(xx)
    data_b = binom(n_B, p_B).pmf(xx)

    ax.bar(xx, data_a, alpha=0.5)
    ax.bar(xx, data_b, alpha=0.5)
    plt.xlabel('converted')
    plt.ylabel('probability')

def plot_ab(p_A, p_B, n_A, n_B):
    d_0 = 0
    d_1 = p_B - p_A
    p_pool = (p_A * n_A + p_B * n_B) / (n_A + n_B)
    SE = np.sqrt(p_pool * (1-p_pool) * (1/n_A + 1/n_B))
    
    xx = np.arange(d_0 - 4*SE, d_1 + 4*SE, 0.0001)
    fig, ax = plt.subplots()
    data_0 = norm.pdf(xx, loc=d_0, scale=SE)
    data_1 = norm.pdf(xx, loc=d_1, scale=SE)

    sns.lineplot(xx, data_0, ax=ax)
    line = ax.get_lines()[-1]
    x,y = line.get_data()
    mask = x > norm.ppf(1-0.05/2, loc=d_0, scale=SE)
    x, y = x[mask], y[mask]
    ax.fill_between(x, y1=y, alpha=0.5, facecolor='blue')

    ax2 = ax.twinx()
    sns.lineplot(xx, data_1, ax=ax2, color='red')
    line = ax2.get_lines()[-1]
    x,y = line.get_data()
    mask = x > norm.ppf(1-0.05/2, loc=d_0, scale=SE)
    x, y = x[mask], y[mask]
    ax2.fill_between(x, y1=y, alpha=0.5, facecolor='darkgreen')
    
    x,y = line.get_data()
    mask_2 = x > norm.ppf(1-0.8, loc=d_1, scale=SE)
    x, y = x[mask_2], y[mask_2]
    ax2.fill_between(x, y1=y, alpha=0.5, facecolor='lightgreen')
    
    calculated_power = 1 - norm.cdf(norm.ppf(1-0.05/2, loc=d_0, scale=SE), loc=d_1, scale=SE)
    
    aaa = 0.15
    ax.annotate(f"Power is: {round(calculated_power, 2)}", xy=(0.1,0.88-aaa), xycoords='figure fraction')
    
def plot_null(p_A, p_B, n_A, n_B):
    d_0 = 0
    d_1 = p_B - p_A
    p_pool = (p_A * n_A + p_B * n_B) / (n_A + n_B)
    #SE = np.sqrt(p_A * (1-p_A) / n_A + p_B * (1-p_B) / n_B)
    SE = np.sqrt(p_pool * (1-p_pool) * (1/n_A + 1/n_B))
    
    xx = np.arange(d_0-3*SE, d_0+3*SE, 0.0001)
    fig, ax = plt.subplots()
    data_0 = norm.pdf(xx, loc=d_0, scale=SE)
    data_1 = norm.pdf(xx, loc=d_1, scale=SE)

    sns.lineplot(xx, data_0, ax=ax)
    line = ax.get_lines()[-1]
    x,y = line.get_data()
    mask = x > norm.ppf(1-0.05/2, loc=d_0, scale=SE)
    x, y = x[mask], y[mask]
    ax.fill_between(x, y1=y, alpha=0.5, facecolor='blue')
    
    line = ax.get_lines()[-1]
    x,y = line.get_data()
    mask = x < norm.ppf(0.05/2, loc=d_0, scale=SE)
    x, y = x[mask], y[mask]
    ax.fill_between(x, y1=y, alpha=0.5, facecolor='blue')
    ax.annotate(f"Min difference needed is: {norm.ppf(1-0.05/2, loc=d_0, scale=SE):.5f}", xy=(0.09,0.89), xycoords='figure fraction')
    
def plot_alt(p_A, p_B, n_A, n_B):
    d_0 = 0
    d_1 = p_B - p_A
    p_pool = (p_A * n_A + p_B * n_B) / (n_A + n_B)
    #SE = np.sqrt(p_A * (1-p_A) / n_A + p_B * (1-p_B) / n_B)
    SE = np.sqrt(p_pool * (1-p_pool) * (1/n_A + 1/n_B))
    
    xx = np.arange(d_1-4*SE, d_1+4*SE, 0.0001)
    fig, ax = plt.subplots()
    data_1 = norm.pdf(xx, loc=d_1, scale=SE)

    sns.lineplot(xx, data_1, ax=ax, color='red')
    line = ax.get_lines()[-1]
    x,y = line.get_data()
    mask_2 = x > norm.ppf(1-0.8, loc=d_1, scale=SE)
    x, y = x[mask_2], y[mask_2]
    ax.fill_between(x, y1=y, alpha=0.5, facecolor='green')
    ax.annotate(f"Min difference is: {norm.ppf(1-0.8, loc=d_1, scale=SE):.5f}", xy=(0.09,0.89), xycoords='figure fraction')

def explore_ab(bcr, lift, size, split):
    plot_ab(bcr, bcr+lift, size/2, size/2)

def calc_sample_size(alpha, beta, p, delta, method):
    """ Based on https://www.evanmiller.org/ab-testing/sample-size.html
    Ref: https://stats.stackexchange.com/questions/357336/create-an-a-b-sample-size-calculator-using-evan-millers-post
    Args:
        alpha (float): How often are you willing to accept a Type I error (false positive)?
        power (float): How often do you want to correctly detect a true positive (1-beta)?
        p (float): Base conversion rate
        pct_mde (float): Minimum detectable effect, relative to base conversion rate.

    """
    if method == 'evanmiller':
        t_alpha2 = norm.ppf(1.0-alpha/2)
        t_beta = norm.ppf(1-beta)

        sd1 = np.sqrt(2 * p * (1.0 - p))
        sd2 = np.sqrt(p * (1.0 - p) + (p + delta) * (1.0 - p - delta))

        return round((t_alpha2 * sd1 + t_beta * sd2) * (t_alpha2 * sd1 + t_beta * sd2) / (delta**2))
    elif method == 'pooled_se':
        """
        References:
            Code taken from Nguyen Ngo: https://towardsdatascience.com/the-math-behind-a-b-testing-with-example-code-part-1-of-2-7be752e1d06f
            Stanford lecture on sample sizes
            http://statweb.stanford.edu/~susan/courses/s141/hopower.pdf
        """
        # standard normal distribution to determine z-values
        standard_norm = scs.norm(0, 1)

        # find Z_beta from desired power
        Z_beta = standard_norm.ppf(1-beta)

        # find Z_alpha
        Z_alpha = standard_norm.ppf(1-alpha/2)

        # average of probabilities from both groups
        pooled_prob = (p + p+delta) / 2

        return 2*(pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2
                 / delta**2)

    
def plot_multiple_alt(d_0, d_1, d_2, SE_0, SE_1, SE_2):
    xx = np.arange(-0.04, 0.06, 0.0001)
    fig, ax = plt.subplots(figsize=(12,6))
    data_0 = norm.pdf(xx, loc=d_0, scale=SE_0)
    data_1 = norm.pdf(xx, loc=d_1, scale=SE_1)
    data_2 = norm.pdf(xx, loc=d_2, scale=SE_2)

    
    aaa = 0.15
    
    sns.lineplot(xx, data_0, ax=ax)
    line = ax.get_lines()[-1]
    x,y = line.get_data()
    mask = x > norm.ppf(1-0.05/2, loc=d_0, scale=SE_0)
    x, y = x[mask], y[mask]
    ax.fill_between(x, y1=y, alpha=0.5, facecolor='blue')
    #ax.axvline(d_0, ls='--')
    ax.annotate(f"Min diff is: {norm.ppf(1-0.05/2, loc=d_0, scale=SE_0):.5f}", xy=(0.1,0.88-aaa), xycoords='figure fraction')
    ax.annotate(f"H0", xy=(d_0,max(data_0)))
    #ax.annotate(f"d0={d_0}", xy=(d_0,0))

    ax2 = ax.twinx()
    sns.lineplot(xx, data_1, ax=ax2, color='green')
    line = ax2.get_lines()[-1]
    x,y = line.get_data()
    mask = x > norm.ppf(1-0.05/2, loc=d_0, scale=SE_0)
    x, y = x[mask], y[mask]
    ax2.fill_between(x, y1=y, alpha=0.5, facecolor='green')
    #ax2.axvline(d_1, color='green', ls='--')
    calculated_power_1 = 1 - norm.cdf(norm.ppf(1-0.05/2, loc=d_0, scale=SE_0), loc=d_1, scale=SE_1)
    ax2.annotate(f"Power group 1 is: {round(calculated_power_1*100)}%", xy=(0.1,0.84-aaa), xycoords='figure fraction')
    ax2.annotate(f"H1", xy=(d_1,max(data_1)))
    #ax2.annotate(f"d1={d_1}", xy=(d_1,0))
    
    ax3 = ax.twinx()
    sns.lineplot(xx, data_2, ax=ax3, color='red')
    line = ax3.get_lines()[-1]
    x,y = line.get_data()
    mask = x > norm.ppf(1-0.05/2, loc=d_0, scale=SE_0)
    x, y = x[mask], y[mask]
    ax3.fill_between(x, y1=y, alpha=0.2, facecolor='red')
    #ax3.axvline(d_2, color='red', ls='--')
    calculated_power_2 = 1 - norm.cdf(norm.ppf(1-0.05/2, loc=d_0, scale=SE_0), loc=d_2, scale=SE_2)
    ax3.annotate(f"Power group 2 is: {round(calculated_power_2*100)}%", xy=(0.1,0.8-aaa), xycoords='figure fraction')
    ax3.annotate(f"H2", xy=(d_2,max(data_2)))
    #ax3.annotate(f"d2={d_2}", xy=(d_2,0))
    

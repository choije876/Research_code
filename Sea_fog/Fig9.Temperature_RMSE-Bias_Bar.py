#
#	Fig9.Temperature_RMSE-Bias_Bar.py
#      - Analyzing T2 & SST 
#
###########################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os

# Font ---------
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False


# Setting ================= 
OBS_name= 'Buoy 값'



# =========================================================
idr = "./"
ifn = idr + f'Temperature_Buoy-GK2A-MODEL_Comparison-donghae.xlsx'

sheet_names = ['동해_SST','동해_T2M']
all_data = []

df_sst = pd.read_excel(ifn, sheet_name=sheet_names[0])
df_t2m = pd.read_excel(ifn, sheet_name=sheet_names[1])


df_sst = df_sst[pd.notna(df_sst['GK2A SST'])].copy()
sst_st = df_sst['GK2A SST']
sst_ct = df_sst['CNTL SST']
sst_sk = df_sst['SKIN SST']
sst_cp = df_sst['CPLD SST']


t2m_bu = df_t2m['Buoy T2M']
t2m_ct = df_t2m['CNTL T2M']
t2m_sk = df_t2m['SKIN T2M']
t2m_cp = df_t2m['CPLD T2M']



# Statistic Analysis ------------
# 1) Calculate RMSE ----
def calculate_rmse(observed, predicted):

    # null 값이 있는 행 제거
    mask = pd.notna(observed) & pd.notna(predicted)
    observed_clean = observed[mask]
    predicted_clean = predicted[mask]

    if len(observed_clean) == 0:
        return np.nan

    return np.sqrt(mean_squared_error(observed_clean, predicted_clean))

sst_rmse_results = {}
t2m_rmse_results = {}
experiments = ['CNTL', 'SKIN', 'CPLD']

sst_rmse_ct = calculate_rmse(sst_st, sst_ct)
sst_rmse_sk = calculate_rmse(sst_st, sst_sk)
sst_rmse_cp = calculate_rmse(sst_st, sst_cp)

t2m_rmse_ct = calculate_rmse(t2m_bu, t2m_ct)
t2m_rmse_sk = calculate_rmse(t2m_bu, t2m_sk)
t2m_rmse_cp = calculate_rmse(t2m_bu, t2m_cp)

sst_rmse_results[experiments[0]] = sst_rmse_ct
sst_rmse_results[experiments[1]] = sst_rmse_sk
sst_rmse_results[experiments[2]] = sst_rmse_cp

t2m_rmse_results[experiments[0]] = t2m_rmse_ct
t2m_rmse_results[experiments[1]] = t2m_rmse_sk
t2m_rmse_results[experiments[2]] = t2m_rmse_cp


# 2) Calculate Bias ----
def calculate_bias(observed, predicted):

    mask = pd.notna(observed) & pd.notna(predicted)
    observed_clean = observed[mask]
    predicted_clean = predicted[mask]

    if len(observed_clean) == 0:
        return np.nan

    return np.mean(predicted_clean - observed_clean)

sst_bias_results = {}
t2m_bias_results = {}

sst_bias_ct = calculate_bias(sst_st, sst_ct)
sst_bias_sk = calculate_bias(sst_st, sst_sk)
sst_bias_cp = calculate_bias(sst_st, sst_cp)

t2m_bias_ct = calculate_bias(t2m_bu, t2m_ct)
t2m_bias_sk = calculate_bias(t2m_bu, t2m_sk)
t2m_bias_cp = calculate_bias(t2m_bu, t2m_cp)

sst_bias_results[experiments[0]] = sst_bias_ct
sst_bias_results[experiments[1]] = sst_bias_sk
sst_bias_results[experiments[2]] = sst_bias_cp

t2m_bias_results[experiments[0]] = t2m_bias_ct
t2m_bias_results[experiments[1]] = t2m_bias_sk
t2m_bias_results[experiments[2]] = t2m_bias_cp




# =====================================================
# Plotting      =======================================
# =====================================================

fts = 10
alpbet = [f"({chr(97+i)})" for i in range(9)]

ymin_t2m, ymax_t2m = -2.5, 2.5
ymin_sst, ymax_sst = -2.5, 2.5  

  
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

sst_rmse_values = list(sst_rmse_results.values())
sst_bias_values = list(sst_bias_results.values())
t2m_rmse_values = list(t2m_rmse_results.values())
t2m_bias_values = list(t2m_bias_results.values())


x = np.arange(len(experiments))
width = 0.35


# 1. RMSE (bar chart) | Bias (line plot) --------------------
# 1-1. SST
ax1  = axes[0]
bars1 = ax1.bar(x-width/2, sst_rmse_values, width, color=['black','black','black'], label='RMSE', edgecolor='black')
bars2 = ax1.bar(x+width/2, sst_bias_values, width, color=['darkred','darkred','darkred'], label='Bias', edgecolor='black')


ax1.set_ylabel('°C', fontsize=10)
ax1.set_ylim([ymin_sst, ymax_sst])
ax1.set_xticks(x)
ax1.set_xticklabels(experiments, rotation=0)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax1.text(0.01, 0.97, f'{alpbet[0]}', transform=ax1.transAxes,
          fontsize=fts, fontweight='bold', va='top', ha='left', color='black')

for bar, value in zip(bars1, sst_rmse_values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{value:.2f}', ha='center', va='bottom')

for bar, value in zip(bars2, sst_bias_values):
    y_pos = bar.get_height() + 0.05 if value >= 0 else bar.get_height() - 0.15
    ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top',
            fontsize=9, color='darkred')



# 1-2.T2m
ax2 = axes[1]
bars3 = ax2.bar(x-width/2, t2m_rmse_values, width, color=['black','black', 'black'], label='RMSE', edgecolor='black')
bars4 = ax2.bar(x+width/2, t2m_bias_values, width, color=['darkred','darkred','darkred'], label='Bias', edgecolor='black')

ax2.set_ylim([ymin_t2m, ymax_t2m])
ax2.set_xticks(x)
ax2.set_xticklabels(experiments, rotation=0)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax2.text(0.01, 0.97, f'{alpbet[1]}', transform=ax2.transAxes,
           fontsize=fts, fontweight='bold', va='top', ha='left', color='black')

for bar, value in zip(bars3, t2m_rmse_values):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{value:.2f}', ha='center', va='bottom', fontsize=9)

for bar, value in zip(bars4, t2m_bias_values):
    y_pos = bar.get_height() + 0.05 if value >= 0 else bar.get_height() - 0.15
    ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top',
            fontsize=9, color='darkred')


ax1.legend(loc='upper right', framealpha=0.9, facecolor='white', frameon=False)


plt.tight_layout()
plt.show()







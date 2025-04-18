import pandas as pd
import numpy as np
import openpyxl
import matplotlib
import matplotlib.pyplot as plt

## TRAINING DATA

# Read data
connect_df = pd.read_csv('widsdatathon2025/TRAIN_NEW/Train_Connectome.csv')
factors_df = pd.read_excel('widsdatathon2025/TRAIN_NEW/Train_Outcome.xlsx')
combined_df = pd.concat([connect_df, factors_df], axis=1)

print(connect_df.shape)
print(factors_df.shape)
print(list(factors_df.columns))
print(combined_df.shape)

# Drop ID column
combined_df.drop(columns=['participant_id'], inplace=True)
print(combined_df.shape)

## Sex Section

# Split dataframe by sex
sex_f = combined_df[combined_df['Sex_F'] == 1]
sex_m = combined_df[combined_df['Sex_F'] == 0]

# Calculate means for each feature
f_means = sex_f.mean()
m_means = sex_m.mean()

# Calculate difference between means and sort
sex_delta = pd.Series(abs(f_means - m_means))
sex_sorted = sex_delta.sort_values(ascending=False)
sex_hist = sex_sorted.drop(['Sex_F', 'ADHD_Outcome'])

# Plot histogram of deltas
plt.hist(sex_hist, density=False, bins=20)  # density=False would make counts
plt.ylabel('Count')
plt.xlabel('Delta')
plt.title('Histogram of edge mean differences ~ Sex')
plt.show()

# Top n values
sex_delta = sex_delta.drop(['Sex_F', 'ADHD_Outcome'])
sex_top_n = sex_delta.nlargest(n=1000, keep='all')
sex_n_edges = list(sex_top_n.index)
print(sex_top_n)

print(combined_df.shape)
sex_train_final = combined_df.loc[:, sex_n_edges]
print(sex_train_final.shape)

# Write to new .csv file
sex_train_final.to_csv('widsdatathon2025/Processed/topn_edges_sex_trn.csv', index=False)

## ADHD Section

# Split dataframe by ADHD status
adhd_pos = combined_df[combined_df['ADHD_Outcome'] == 1]
adhd_neg = combined_df[combined_df['ADHD_Outcome'] == 0]

# Calculate means for each feature
pos_means = adhd_pos.mean()
neg_means = adhd_neg.mean()

# Calculate difference between means and sort
adhd_delta = abs(pos_means - neg_means)
adhd_sorted = adhd_delta.sort_values(ascending=False)
adhd_hist = adhd_sorted.drop(['Sex_F', 'ADHD_Outcome'])

# Plot histogram of deltas
plt.hist(adhd_hist, density=False, bins=20)  # density=False would make counts
plt.ylabel('Count')
plt.xlabel('Delta')
plt.title('Histogram of edge mean differences ~ ADHD')
plt.show()

# Top n values
adhd_delta = adhd_delta.drop(['Sex_F', 'ADHD_Outcome'])
adhd_top_n = adhd_delta.nlargest(n=1000, keep='all')
adhd_n_edges = list(adhd_top_n.index)
print(adhd_top_n)

print(combined_df.shape)
adhd_train_final = combined_df.loc[:, adhd_n_edges]
print(adhd_train_final.shape)

# Write to new .csv file
adhd_train_final.to_csv('widsdatathon2025/Processed/topn_edges_adhd_trn.csv', index=False)

## TESTING DATA

# Read Testing data
connect_df_test = pd.read_csv('widsdatathon2025/TEST/Test_Connectome.csv')
print(connect_df_test.shape)

# Drop ID column
connect_df_test.drop(columns=['participant_id'], inplace=True)
print(connect_df_test.shape)

## Slice by top edges for sex factor
sex_test_final = connect_df_test.loc[:, sex_n_edges]
print(sex_test_final.shape)

# Write to new .csv file
sex_test_final.to_csv('widsdatathon2025/Processed/topn_edges_sex_tst.csv', index=False)

## Slice by top edges for ADHD factor
adhd_test_final = connect_df_test.loc[:, adhd_n_edges]
print(adhd_test_final.shape)

# Write to new .csv file
adhd_test_final.to_csv('widsdatathon2025/Processed/topn_edges_adhd_tst.csv', index=False)

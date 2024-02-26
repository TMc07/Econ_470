import pandas as pd
from git import Repo
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import seaborn as sns
import warnings

# Suppress FutureWarnings from NumPy
warnings.simplefilter(action='ignore', category=FutureWarning)

directory_path = r"C:\Users\tymcc\OneDrive\Desktop\470_Project\Mexico_Project\Graphs"

# Clone the repository
repo_url = "https://github.com/TMc07/Econ_470.git"

# Define the path to the CSV files
local_dir = "Econ_470"  # Existing directory with the cloned repository
concen_path = 'Concen.csv'
hogares1_path = 'hogares1.csv'
Poblacion_path = 'Poblacion.csv'

# Load the CSV files
concen_df = pd.read_csv(concen_path)
hogares1_df = pd.read_csv(hogares1_path)
Poblacion_df = pd.read_csv(Poblacion_path)
# Merge the dataframes on 'folio'
merged_df = pd.merge(concen_df, hogares1_df, on='folio')
merged_df2 = pd.merge(merged_df, Poblacion_df, on='folio')
# Save the merged dataframe to a new CSV file
merged_df2.to_csv("merged_data2.csv", index=False)

shortened = merged_df2.head()
shortened.to_csv("shortened.csv", index=False)

merged_df_cleaned = merged_df2.dropna()
merged_df_cleaned.to_csv("merged_df_cleaned.csv", index=False)

# Define the list of variables to keep
variables_to_keep = ['folio', 'edad_y', 'beca', 'ed_formal', 'edocony', 'tot_resi', 'p0a11', 'ingtot', 'medica', 'medica_binned']

#Binning of Medica into 100 bins using quartiles
merged_df2['medica_binned'] = pd.qcut(merged_df2['medica'], q=100, labels=False, duplicates='drop')

# Create a new DataFrame containing only the specified variables
df_subset = merged_df2[variables_to_keep]
df_Scholarship = df_subset[df_subset['beca'] == '6']

# Display the new DataFrame
df_subset.to_csv("df_subset.csv", index=False)
df_Scholarship.to_csv("df_Scholarship.csv", index=False)

# Load the dataset
df = pd.read_csv('df_subset.csv')

# Trim whitespace and replace empty strings or spaces in 'beca' with NaN
df['beca'] = df['beca'].str.strip().replace('', np.nan).replace(' ', np.nan)
df['edocony'] = df['edocony'].str.strip().replace('', np.nan).replace(' ', np.nan)

# Convert 'beca' to numeric
df['beca'] = pd.to_numeric(df['beca'], errors='coerce')

# Drop rows where 'beca' is NaN
df.dropna(subset=['beca'], inplace=True)
df.dropna(subset=['edocony'], inplace=True)

# Select the treatment and covariates
treatment = 'beca'
covariates = ['ed_formal', 'edocony', 'tot_resi', 'p0a11', 'edad_y']

# Define treatment: 1 if 'beca' equals 6, 0 otherwise
df['treatment'] = np.where(df['beca'] == 6, 1, 0)


# Create a logistic regression model for propensity score estimation
logistic = LogisticRegression()
propensity_score = logistic.fit(df[covariates], df['treatment']).predict_proba(df[covariates])[:, 1]
df['propensity_score'] = propensity_score


# Perform nearest neighbor matching
causal = CausalModel(
    Y=df['ingtot'].values, 
    D=df['treatment'].values, 
    X=df['propensity_score'].values
)
causal.est_via_matching(bias_adj=True)

# Assess balance by comparing means and distributions
print("Balance Table Before Matching With no Normal Adjustments:")
print(causal.summary_stats)

print("\nBalance Table After Matching With no Normal Adjustments:")
print(causal.estimates)

# Create a copy of the DataFrame to hold standardized values
df_standardized = df.copy()

# Standardize covariates
scaler = StandardScaler()
df_standardized[covariates] = scaler.fit_transform(df[covariates])

# Create a logistic regression model for propensity score estimation
logistic = LogisticRegression()
propensity_score = logistic.fit(df_standardized[covariates], df_standardized['treatment']).predict_proba(df_standardized[covariates])[:, 1]
df_standardized['propensity_score'] = propensity_score

# Perform nearest neighbor matching
causal = CausalModel(
    Y=df_standardized['ingtot'].values, 
    D=df_standardized['treatment'].values, 
    X=df_standardized['propensity_score'].values
)
causal.est_via_matching(bias_adj=True)

# Assess balance by comparing means and distributions

print("\nBalance Table After Matching:")
print(causal.estimates)

# Logistic regression for treatment
logit_model = sm.Logit(df_standardized['treatment'], sm.add_constant(df_standardized[covariates]))
logit_result = logit_model.fit()
print("\nLogistic Regression for Treatment:\n")
print(logit_result.summary())

# Linear regression for outcome
linear_model = sm.OLS(df_standardized['ingtot'], sm.add_constant(df_standardized[covariates + ['treatment']]))
linear_result = linear_model.fit()
print("\nLinear Regression for Outcome:\n")
print(linear_result.summary())

def save_propensity_score_plots(df, treatment, output_dir):
    # Set style for seaborn
    sns.set(style="whitegrid")

    # Histogram/Density Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df[treatment] == 1]['propensity_score'], color="skyblue", label='Treatment', kde=True)
    sns.histplot(df[df[treatment] == 0]['propensity_score'], color="red", label='Control', kde=True)
    plt.title('Propensity Score Distribution by Treatment Group')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{output_dir}/Propensity_Score_Distribution.png")
    plt.close()

def save_propensity_score_scatter_plot(df, treatment, medica_col, output_dir):
    # Scatter Plot with 'medica' as y-axis
    plt.figure(figsize=(10, 6))

    # Plot treatment group
    plt.scatter(df[df[treatment] == 1]['propensity_score'], 
                df[df[treatment] == 1][medica_col], 
                alpha=0.2, color="skyblue", label='Treatment')

    # Plot control group
    plt.scatter(df[df[treatment] == 0]['propensity_score'], 
                df[df[treatment] == 0][medica_col], 
                alpha=0.2, color="red", label='Control')

    plt.title('Propensity Score vs. Medica by Treatment Group')
    plt.xlabel('Propensity Score')
    plt.ylabel(medica_col)
    plt.legend()
    plt.savefig(f"{output_dir}/Propensity_Score_vs_Medica_Scatter.png")
    plt.close()

# Usage
save_propensity_score_plots(df, 'treatment', directory_path)
save_propensity_score_scatter_plot(df, 'treatment', 'medica', directory_path)

# Log-transform only the 'ingtot' variable
df['ingtot_log'] = np.log(df['ingtot'] + 1)  # Adding 1 to avoid log(0)

# Check and ensure that 'treatment' column exists in your DataFrame
if 'treatment' in df.columns:
    # Define 'medica_prop' before filtering and plotting
    df['medica_prop'] = df['medica'] / df['ingtot']
    df = df[df['medica'] != 0]  # Removing 'medica' values of 0


    # Separate assignments for Oportunidades and Seguro Popular
    df['assigned_to_oportunidades'] = df['beca'] == 6
    df['assigned_to_seguro_popular'] = df['tot_resi'] > 5

    # Filtering for different groups
    oportunidades_df = df[df['assigned_to_oportunidades']]
    seguro_popular_df = df[df['assigned_to_seguro_popular']]

    # Calculating 'medica_prop' for both groups
    oportunidades_df.loc[:, 'medica_prop'] = oportunidades_df['medica'] / oportunidades_df['ingtot']
    seguro_popular_df.loc[:, 'medica_prop'] = seguro_popular_df['medica'] / seguro_popular_df['ingtot']


    # Plotting the proportion of 'medica' vs. logged 'ingtot' for both groups
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='ingtot_log', y='medica_prop', data=oportunidades_df, label='Oportunidades')
    sns.lineplot(x='ingtot_log', y='medica_prop', data=seguro_popular_df, label='Seguro Popular')
    plt.title('Medica Proportion vs. Logged Income for Oportunidades and Seguro Popular Groups')
    plt.xlabel('Logged Income (ingtot)')
    plt.ylabel('Proportion of Medica')
    plt.legend()
    plt.grid(True)
else:
    print("'treatment' column not found in the DataFrame.")

# Summary statistics for 'medica_prop' in Oportunidades group
oportunidades_summary = oportunidades_df['medica_prop'].describe()

# Summary statistics for 'medica_prop' in Seguro Popular group
seguro_popular_summary = seguro_popular_df['medica_prop'].describe()

# Separate data for 'assigned_to_oportunidades' and 'assigned_to_seguro_popular'
oportunidades_data = df[df['assigned_to_oportunidades']]
seguro_popular_data = df[df['assigned_to_seguro_popular']]

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ingtot_log', y='medica_prop', data=oportunidades_data, color='blue', label='Oportunidades')
sns.scatterplot(x='ingtot_log', y='medica_prop', data=seguro_popular_data, color='red', label='Seguro Popular')
plt.xlabel('Log of Total Income (ingtot_log)')
plt.ylabel('Medical Proportion (medica_prop)')
plt.ylim(0, 0.1)  # Keeping the y-axis limit
plt.xlim(6, 14)
plt.title('Comparison of Medical Proportion vs Total Income')
plt.legend()

# Save the plot
plt.savefig('medical_proportion_vs_income.png')  # Adjust the file path as needed

# Separate data for 'assigned_to_oportunidades' and 'assigned_to_seguro_popular'
oportunidades_data = df[df['assigned_to_oportunidades']]['ingtot_log']
seguro_popular_data = df[df['assigned_to_seguro_popular']]['ingtot_log']

# Plotting the cumulative distribution
plt.figure(figsize=(10, 6))
sns.ecdfplot(oportunidades_data, label='Oportunidades', color='blue')
sns.ecdfplot(seguro_popular_data, label='Seguro Popular', color='red')
plt.xlabel('Medical Proportion (ingtot_log)')
plt.ylabel('Cumulative Distribution')
plt.title('Cumulative Distribution of Medical Proportion')
plt.legend()

# Save the plot
plt.savefig('medical_proportion_cumulative_distribution.png')  # Adjust the file path as needed


def save_medica_prop_cumulative_distribution(df, output_dir):
    # Setting seaborn style
    sns.set(style="whitegrid")

    # Separate data for 'assigned_to_oportunidades' and 'assigned_to_seguro_popular'
    oportunidades_data = df[df['assigned_to_oportunidades']]['ingtot_log']
    seguro_popular_data = df[df['assigned_to_seguro_popular']]['ingtot_log']

    # Plotting the cumulative distribution
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(oportunidades_data, label='Oportunidades', color='blue')
    sns.ecdfplot(seguro_popular_data, label='Seguro Popular', color='red')
    plt.xlabel('Medical Proportion (ingtot_log)')
    plt.ylabel('Cumulative Distribution')
    plt.title('Cumulative Distribution of Medical Proportion')
    plt.legend()


    # Save the plot to the specified directory
    plt.savefig(f"{output_dir}/Medical_Proportion_Cumulative_Distribution.png")
    plt.close()

# Usage
# Assuming 'df' is your DataFrame and 'directory_path' is the path to the directory where you want to save the plot
save_medica_prop_cumulative_distribution(df, directory_path)


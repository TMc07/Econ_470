#This is the Repo that holds the code and csv files needed to recreate the project. The file Final_form.py holds the python script to run the analysis and it returns the regression results in the terminal that are similar to the paper (the paper was published in mulitple parts and the variables used in the construction of the graphs are built based off these regression results).

After this, the graphs hold the [alt text](Graphs/Medical_Proportion_Cumulative_Distribution.png) which is the graph that lines up from the paper and models the key results of the paper. In this the paper found that while there are differences on the extremes of the policies, neither of the policies are addressing different components of poverty and both help to reduce the expenses being placed onto lower income residents in Mexico. 



This is the prompt that was used that should be able to be fed into chatgpt and return the script in "Final_form.py"

"I'm working on a data analysis project in Python involving economic data from Mexico. The project includes data preprocessing, statistical modeling, causal inference, and visualization. I have three CSV files: 'Concen.csv', 'hogares1.csv', and 'Poblacion.csv', which need to be merged based on a common column 'folio'. After merging, I need to clean the data, handle missing values, and create a subset of variables. Here are the specific requirements:

Data Loading and Merging:

Load the CSV files using Pandas.
Merge the datasets on the 'folio' column.
Create a cleaned version by dropping missing values.
Data Transformation:

Perform binning on the 'medica' variable into 100 bins using quartiles.
Select a subset of variables for further analysis.
Propensity Score Estimation:

Implement logistic regression to estimate propensity scores for a treatment variable, 'beca'.
Use the propensity scores for nearest neighbor matching in causal analysis.
Causal Inference:

Use the CausalModel from the 'causalinference' package.
Perform matching and assess the balance before and after matching.
Data Standardization:

Standardize the covariates before rerunning the logistic regression.
Advanced Statistical Analysis:

Run logistic and linear regression models on the standardized data.
Interpret the results.
Data Visualization:

Generate histograms and scatter plots to visualize the propensity scores and other relationships.
Use seaborn and matplotlib for plotting.
Saving Outputs:

Save various transformed and subset data frames to CSV files.
Save all plots to a specific directory.
Could you provide a Python script to accomplish these tasks?"
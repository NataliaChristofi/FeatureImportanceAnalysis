#Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import plotly.io as pio
import plotly.graph_objects as go
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Where to save the figures
PROJECT_ROOT_DIR = "/project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

#printing all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Suppressing warnings
import warnings
warnings.simplefilter(action = "ignore")


# --------------------------------------------------------------------

## FUNCTION TO NORMALIZE THE DataFrame
def normalize_func(df_train, df_test):
    #create a variable for the scaler
    scaler = MinMaxScaler()
    #obtain the column names of the dataframe as a list
    df_train_columns = df_train.columns.tolist()
    df_test_columns = df_test.columns.tolist()
    
    #apply .fit_transform to the train dataset
    df_train[df_train_columns] = scaler.fit_transform(df_train[df_train_columns])
    #apply .transform to the test dataset
    df_test[df_test_columns] = scaler.transform(df_test[df_test_columns])



## FUNCTION TO ADD THE SCORED INTO df_evaluation_score SO IT CAN BE EASIER TO USE
def adding_to_df(df, col1_value, col2_value, col3_value):
    data = pd.DataFrame({"Model": col1_value,
                         "Evaluation_Name": col2_value,
                         "score": col3_value },
                       index=[0])
    df = df.append(data,ignore_index=True )
    return(df)
    
    

## FUNCTION TO SAVE THE FIGURES
def save_fig(fig_id, tight_layout=True, fig_extension="jpeg", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    #if tight_layout:
     #   plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    


# def feature_importance(model, train_features):
#     # get importance
#     importance = model.coef_
#     # summarize feature importance
#     for i,v in enumerate(importance):
#         #print('Feature: %0d, Score: %.5f' % (i,v))
#         print(f"\033[1m Feature {i}: {train_features.columns.tolist()[i]}\033[0m,\n Score: {round(v,5)}")
#         # plot feature importance
#     plt.bar([x for x in range(len(importance))], importance)
#     plt.show()

    
## FUNCTION TO DISPLAY THE COEFFICIENTS VALUES PER FEATURE AND THEN PLOT THEM IN A HORTIZONTAL BAR CHART
def feature_importance(model, train_features):
    # get importance
    importance = model.coef_
    # summarize feature importance
    for i,v in enumerate(importance):
        #print('Feature: %0d, Score: %.5f' % (i,v))
        print(f"\033[1m Feature {i}: {train_features.columns.tolist()[i]}\033[0m,\n Score: {round(v,5)}")
        # plot feature importance in a horizontal bar chart where negative values will be red and positive green
    coefs = pd.DataFrame(
        model.coef_,
        columns=['Coefficients'], index=train_features.columns
    )
    model_name = type(model).__name__
    coefs = coefs['Coefficients']
    color = (coefs > 0).apply(lambda x: 'g' if x else 'r')    
    coefs.plot(kind='barh', figsize=(9, 7),
              color=color)
    #the plot title will be different each time based on the model
    plt.title(f'{model_name} model')
    plt.xlabel('Coefficient Values', labelpad=10)
    plt.ylabel('Features Names', labelpad=10)
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)
    plt.show()
    
    
    
## FUNCTION TO DISPLAY THE COEFFICIENTS VALUES PER FEATURE FOR TREE MODELS
def feature_importance_trees(model, train_features):
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        #print('Feature: %0d, Score: %.5f' % (i,v))
        print(f"\033[1m Feature {i}: {train_features.columns.tolist()[i]}\033[0m,\n Score: {round(v,5)}")
        # plot feature importance
    coefs = pd.DataFrame(
        model.feature_importances_,
        columns=['Coefficients'], index=train_features.columns
    )
    model_name = type(model).__name__
    coefs = coefs['Coefficients']
    coefs = coefs.sort_values()
    color = ['grey' if (x < max(coefs)) else 'red' for x in coefs ]
    coefs.plot(kind='barh', figsize=(9, 7),
               color=color)
    #the plot title will be different each time based on the model
    plt.title(f'{model_name} model')
    plt.xlabel('Coefficient Values', labelpad=10)
    plt.ylabel('Features Names', labelpad=10)
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)
    plt.show() 



# def feature_importance_trees(model, train_features):
#     # get importance
#     importance = model.feature_importances_
#     # summarize feature importance
#     for i,v in enumerate(importance):
#         #print('Feature: %0d, Score: %.5f' % (i,v))
#         print(f"\033[1m Feature {i}: {train_features.columns.tolist()[i]}\033[0m,\n Score: {round(v,5)}")
#         # plot feature importance
#     plt.bar([x for x in range(len(importance))], importance)
#     plt.show() 
    

## FUNCTION TO GET THE NAME OF A DATAFRAME
# def get_df_name(df):
#     name =[x for x in globals() if globals()[x] is df][0]
#     return name    
    

## FUNCTION TO PLOT A SCATTER PLOT BETWEEN TWO VARIABLES
def scatter_plot(df,x_col1,y_col2):
    plt.scatter(df[x_col1], df[y_col2], color='red')
    plt.title(y_col2 + ' Vs ' + x_col1, fontsize=14)
    plt.xlabel(x_col1, fontsize=14)
    plt.ylabel(y_col2, fontsize=14)
    plt.grid(True)
    plt.show();
    
    
## FUNCTION TO MAKE BARPLOTS TO SEE THE DISTRIBUTION OF THE VARIABLES
def bar_plot(df,fig_name):
    df_columns_list = df.columns.tolist()
    bar_plot = df[df_columns_list]
    fig = plt.figure(figsize =(10, 7))
    # Creating plot
    plt.boxplot(bar_plot)
    #plt.title(f'Bar Plot {df_name}')
    #save fig
    save_fig(fig_name)
    #show plot
    plt.show();
    
    
## FUNCTION TO MAKE PAIRPLOTS WITH SEABORN TO SEE THE TREND     
def pairplot(train_dataset, initial_df, x, y):
    x_vars = train_dataset.columns.tolist()
    g = sns.pairplot(initial_df, x_vars=x_vars[x:y],
                 y_vars="Incidents_per_1000", 
                 kind='reg', plot_kws={'line_kws':{'color':'red'}},
                 height=4)
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    
# --------------------------------------------------------------------

### DATA WRANGLING FUNCTIONS 

#Checking share of missing and unique values to determine columns to drop if any 
def share_missing_values(df):
    share_missing_vals_num = (df.isnull().sum() / len(df))*100
    share_missing_vals_str = round(share_missing_vals_num,2).astype(str)+"%"
    share_missing_unique_vals_df = pd.DataFrame({"share_of_missing_values": share_missing_vals_str, "share_of_missing_values_num": share_missing_vals_num})
    share_missing_unique_vals_df['count_of_unique_values'] = df.apply(lambda row: row.nunique(dropna = True)).to_frame()
    share_missing_unique_vals_df = share_missing_unique_vals_df.sort_values(by = "share_of_missing_values_num", ascending = False)
    share_missing_unique_vals_df = share_missing_unique_vals_df.drop("share_of_missing_values_num", axis = 1)
    return share_missing_unique_vals_df
    

#Visualizing missing values
def viz_missing_values(df):
    #Creating plot architecture
    fig, ax = plt.subplots(figsize = (12, 6))
    #Plotting
    colors = ["#d3d3d3", "#9BBAE2"]
    columns = [name.replace("_"," ").capitalize() for name in df.columns]
    ax = sns.heatmap(df.isnull(), 
                     cmap = sns.color_palette(colors), 
                     cbar_kws = dict(
                     orientation = "vertical", 
                     shrink = 0.1,
                     aspect = 2.0))
    #Setting parameters
    ax.set_yticks(range(0,len(df),500000))
    ax.set_yticklabels(range(0,len(df),500000))
    ax.set_ylabel("Number of observations")
    ax.set_xticklabels(columns)
    ax.vlines(range(len(df.columns)+1), 
              ymin = 0, 
              ymax = len(df)-1, 
              color = "white", 
              linewidth = 2)
    ax.set_title("Missing values per column",
                 color = "#696969",
                 weight = "bold",
                 size = 12, 
                 pad = 10.0)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(["Value available", "Value missing"])
    plt.tick_params(bottom = False, left = False)
    plt.show();
    
# --------------------------------------------------------------------    
    
### PRE PROCESSING FUNCTIONS

## function to get the values for the past 120 days 
def rolling_120d(df):
    df_columns_list = df.columns.tolist()
    #only taking from the 4th column onwards as we don't need Rig, month and year
    df_columns_list = df_columns_list[3:]
    for i in df_columns_list:
        df[f"{i}_120d"]=(df
                         .groupby('Rig')[i]
                         .rolling(min_periods=3, window=3, closed= "left")
                         .sum().reset_index(0,drop=True))
    return df.drop(df_columns_list, axis=1)
    
    

## function to get the share of the risk type one hot encoded column
def share_risk_type(df, column_share):
    df[f"{column_share}_percentage"] = (
    df[column_share] / (
        (df["Evaluation Only_120d"]+df["First Aid Incident_120d"]
         + df["Medical Treatment Only_120d"]+df["No IADC Injury Classification_120d"]
         + df["Restricted Work Incident_120d"])))
    return df
    
## function to get the share of the crew category one hot encoded column
def share_crew_category(df):
    crew_cat_columns = ['Catering_120d','Noble_Expat_120d','Noble_National_120d',
                        'Noble_Personnel_120d','Noble_Services_120d',
                        'Cat_Type_Operator_120d','Operator_Operations_120d',
                        'Operator_Services_120d','Other_120d']
    for i in crew_cat_columns:
        df[f"{i}_percentage"] = (
            df[i] / (
                (df["Catering_120d"]+ df["Noble_Expat_120d"]
                 + df["Noble_National_120d"]+ df["Noble_Personnel_120d"]
                 + df["Noble_Services_120d"] + df["Cat_Type_Operator_120d"]
                 + df["Operator_Operations_120d"]+ df["Operator_Services_120d"]
                 + df["Other_120d"])))
        
    return df.drop(crew_cat_columns, axis=1)
    
## function to get the share of the risk level one hot encoded column
def share_risk_level(df, column_share):
    df[f"{column_share}_percentage"] = (
    df[column_share] / (
        (df["High_risk_120d"]
         +df["Low_risk_120d"]
         +df["Medium_risk_120d"])))
    return df
    
## function to get the total alpha and beta values per month
def get_count_injury(df, cat_column):   
    count_injury = df.groupby(["Rig","year","year_month"])[cat_column].sum()
    count_injury = pd.Series(count_injury).reset_index(drop = True)
    return count_injury


##Function for correlation matrix
def corrMatrix(df, fig_name):
    corrMatrix = df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corrMatrix, annot=True, linewidths=.5)
    #save fig
    save_fig(fig_name)
    plt.show();
    
    
    
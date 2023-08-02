import streamlit as st
import pandas as pd
from PIL import Image  
import pandas as pd
import os
import sqlalchemy
import pickle
from datetime import datetime
import numpy as np
import xgboost as xg
from plotly_calplot import calplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from streamlit_option_menu import option_menu


@st.cache_resource
def get_connection():
    # execute to ensure local proxy variables not used
    for k in ['HTTP_PROXY', 'HTTPS_PROXY']:
        os.environ.pop(k, None)
        os.environ.pop(k.lower(), None)

    engine= sqlalchemy.create_engine(
        os.environ['snowflake_conn'],
        execution_options=dict(autocommit=True)
    )
    return engine


@st.cache_data
def fetch_flight_data(fltnumber):
    with st.spinner('Loading Data...'):
        time.sleep(0.5)
        flight_data = pd.read_sql(f"SELECT * FROM DEMODATA WHERE fltnum = '{fltnumber}' AND cmp = 'Y' ORDER BY fltdep ASC", get_connection())
    return flight_data

@st.cache_data
def fetch_classflight_data(fltnumber):
    with st.spinner('Loading Data...'):
        time.sleep(0.5)
        flight_data = pd.read_sql(f"SELECT * FROM trainingdata WHERE fltnum = '{fltnumber}' AND cmp = 'Y' AND fltdep >= '2019-01-01' ORDER BY fltdep ASC", get_connection())
    return flight_data


img = Image.open("emirates_logo.png")
img2 = Image.open("graphics.png")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 250px;
        max-width: 500px;
    }
    """,
    unsafe_allow_html=True,
)   

st.sidebar.title("SODP prediction")
st.sidebar.image(img2,width = 100)
side_bar = st.sidebar.radio('What would you like to view?', [ 'Classification Model', '📝Help: CM', 'Regression Model', '📝Help: RM'])

if side_bar == 'Classification Model':
    header = st.container()
    features = st.container()

    with header:
        
        text_col,image_col = st.columns((5.5,1))
        
        with text_col:
            st.title("SODP Classification Model")
            st.markdown("Classifies whether the flight will get full **30 days** prior to departure")
        
        with image_col:
            st.write("##")
            st.image("https://media.giphy.com/media/gKqWZbwmP4I3SwUof9/giphy.gif", width = 150)

    with features:
        st.markdown("Select the folowing:")
        col1, col2 = st.columns((5,5))
        with col1:
            df1_new = pd.read_sql("SELECT DISTINCT lego FROM trainingdata ORDER BY lego", get_connection())
            option_1 = df1_new['lego'].tolist()
            default_option = option_1.index('BOM')
            selected_1 = st.selectbox('Select Leg Origin', option_1, index = default_option)
        with col2:
            df2_new = pd.read_sql(f"SELECT DISTINCT legd FROM trainingdata WHERE lego = '{selected_1}' ORDER BY legd", get_connection())
            option_2 = df2_new['legd'].tolist()
            selected_2 = st.selectbox('Select Leg Destination', option_2)

        df3_new = pd.read_sql(f"SELECT DISTINCT fltnum FROM trainingdata WHERE lego = '{selected_1}' AND legd = '{selected_2}' ORDER BY fltnum", get_connection())
        option_3 = df3_new['fltnum'].tolist()
        selected_3 = st.selectbox('Select Flight Number ', option_3)
        
        st.write("You have selected:")
        st.write('Origin:', selected_1, ", Destination: ", selected_2, ", Flt num: ", selected_3)
    
        if st.button("Generate"):
            with open(f"xgboost_model_{selected_3}.pickle", "rb") as f:
                model = pickle.load(f)
        
            flight = fetch_classflight_data(selected_3)
            flight['fltdep'] = pd.to_datetime(flight['fltdep']).dt.date
            flight['snap'] = pd.to_datetime(flight['snap']).dt.date
            flight['bkd'] = flight['bkd'] + flight['grp_bkd']
            flight = flight.drop('grp_bkd', axis = 1)
            flight['sodp'] = np.where(flight['bkd'] > flight['cap'], (flight['fltdep'] - flight['snap']).dt.days, 0)
            flight['max_sodp'] = flight.groupby('fltdep')['sodp'].transform('max')
            flight['sodp'] = flight['max_sodp']
            flight.drop(columns=['max_sodp'], inplace=True)
            
            flight.loc[:, '30days-prior'] = flight['sodp'].apply(lambda x: 'True' if x >= 30 else 'False')
            flight['30days-prior'] = flight['30days-prior'].map({'False':0,'True':1})
            flight = flight.loc[flight.groupby('fltdep')['30days-prior'].idxmax()]
            
            #TESTING ONLY
            target_date = pd.to_datetime('2019-01-01')
            distinct_flights2 = flight[flight['fltdep'] >= target_date]
            date_df_temp2 = distinct_flights2
            date_df_temp2.fillna(0, inplace=True)
            
            date_df_temp2['fltdep'] = pd.to_datetime(date_df_temp2['fltdep'])
            date_df_temp2['day'] = date_df_temp2['fltdep'].dt.day
            date_df_temp2['month'] = date_df_temp2['fltdep'].dt.month
            date_df_temp2['year'] = date_df_temp2['fltdep'].dt.year
            date_df_temp2['day_of_week'] = date_df_temp2['fltdep'].dt.dayofweek
            
            for_plotting = date_df_temp2[['fltdep','30days-prior']]
            # Step 3: One-Hot Encoding (if needed)
            date_df_temp2 = pd.get_dummies(date_df_temp2, columns=['month', 'day_of_week'])
            date_df_temp2 = date_df_temp2.drop(['fltnum','lego','legd','fltdep','cmp','days_prior'], axis = 1)
            
            test_data = date_df_temp2
            test_data = test_data.drop(['snap'], axis = 1)    
            
            testing = test_data.drop(['30days-prior'], axis = 1)
            
            feature_names = model.get_booster().feature_names
            for col in feature_names:
                if col not in testing.columns:
                    testing[col] = 0
                    
            testing = testing[feature_names]
            
            pred = model.predict(testing)
            act = test_data['30days-prior']
            
            #PREDICTED
            start_date = '2019-01-01'
            end_date = '2019-12-31'
            full_date_range = pd.date_range(start=start_date, end=end_date)
            date_df = pd.DataFrame({'Date': full_date_range})
            date_df['Date'] = pd.to_datetime(date_df['Date'])
            
            for_plotting1 = for_plotting
            for_plotting1['30days-prior'] = pred

            merged_df1 = pd.merge(date_df, for_plotting1, left_on='Date', right_on='fltdep', how='left')
            
            merged_df1['30days-prior'].fillna(0, inplace=True)
            merged_df1['30days-prior'] = merged_df1['30days-prior'].astype(int)

            merged_df1['Date'] = pd.to_datetime(merged_df1['Date'], format='%Y-%m-%d')
            max_value = merged_df1['30days-prior'].max()
            
            fig = calplot(
                merged_df1, 
                x= "Date", 
                y = "30days-prior",
                dark_theme=False,
                years_title=True,
                gap=1,
                name="30days-prior",
                month_lines_width=2, 
                month_lines_color="black",
                colorscale="magma",
                showscale=True,
                cmap_max=1,
                cmap_min= 0
                )

            merged_df1['Date'] = pd.to_datetime(merged_df1['Date'])
            date_formatted2 = merged_df1['Date'].dt.strftime('%Y-%m-%d')

            hover_template = (
                '<b>Date</b>: %{customdata[0]}<br>' +
                '<b>SODP</b>: %{customdata[1]:,.0f}<br>'
                '<extra></extra>'
            )

            customdata = np.stack((date_formatted2, merged_df1['30days-prior']), axis=-1)

            fig.update_traces(hovertemplate=hover_template, customdata=customdata)
            fig.update_layout(title_text=f'<b>Predicted 30days-prior for flight {selected_3}</b>')
            fig.update_layout(width=800)


            st.plotly_chart(fig)
            
            st.write("")
            #ACTUAL
            start_date = '2019-01-01'
            end_date = '2019-12-31'
            full_date_range = pd.date_range(start=start_date, end=end_date)
            date_df2 = pd.DataFrame({'Date': full_date_range})
            date_df2['Date'] = pd.to_datetime(date_df2['Date'])
            
            for_plotting2 = for_plotting
            for_plotting2['30days-prior'] = act

            merged_df2 = pd.merge(date_df, for_plotting2, left_on='Date', right_on='fltdep', how='left')
            
            # Step 6: Fill missing values in 'Value' with 2 
            merged_df2['30days-prior'].fillna(0, inplace=True)
            merged_df2['30days-prior'] = merged_df2['30days-prior'].astype(int)

            # Add a Date column to the date_df_temp DataFrame
            merged_df2['Date'] = pd.to_datetime(merged_df2['Date'], format='%Y-%m-%d')

            fig = calplot(
                merged_df2, 
                x= "Date", 
                y = "30days-prior",
                dark_theme=False,
                years_title=True,
                gap=1,
                name="30days-prior",
                month_lines_width=2, 
                month_lines_color="black",
                colorscale="magma",
                showscale=True,
                cmap_max=1,
                cmap_min= 0
                )

            merged_df2['Date'] = pd.to_datetime(merged_df2['Date'])

            date_formatted2 = merged_df2['Date'].dt.strftime('%Y-%m-%d')

            # Create a hover template to customize the tooltip content
            hover_template = (
                '<b>Date</b>: %{customdata[0]}<br>' +
                '<b>SODP</b>: %{customdata[1]:,.0f}<br>'
                '<extra></extra>'
            )

            # Prepare custom data by stacking the converted 'Date' column and other columns
            customdata = np.stack((date_formatted2, merged_df2['30days-prior']), axis=-1)

            # Update the hovertemplate for the figure and add custom data
            fig.update_traces(hovertemplate=hover_template, customdata=customdata)
            fig.update_layout(title_text=f'<b>Actual 30days-prior for flight {selected_3}</b>')
            fig.update_layout(width=800)
            st.plotly_chart(fig)
            
            st.write("")
            
            #Difference
            start_date = '2019-01-01'
            end_date = '2019-12-31'
            full_date_range = pd.date_range(start=start_date, end=end_date)
            date_df3 = pd.DataFrame({'Date': full_date_range})
            date_df3['Date'] = pd.to_datetime(date_df3['Date'])

            # Step 5: Combine the current testing data with the date_df
            date_df3['30days-prior'] = merged_df2['30days-prior'] - merged_df1['30days-prior']
            date_df_temp3 = date_df3

            # Step 6: Fill missing values in 'Value' with 2 
            date_df_temp3['30days-prior'].fillna(2, inplace=True)
            date_df_temp3['30days-prior'] = date_df_temp3['30days-prior'].astype(int)

            # Add a Date column to the date_df_temp DataFrame
            date_df_temp3['Date'] = pd.to_datetime(date_df_temp3['Date'], format='%Y-%m-%d')

            fig = calplot(
                date_df_temp3, 
                x= "Date", 
                y = "30days-prior",
                dark_theme=False,
                years_title=True,
                gap=1,
                name="30days-prior",
                month_lines_width=0.5, 
                month_lines_color="black",
                colorscale="RdBu",
                showscale=True,
                cmap_max = 1,
                cmap_min = - 1
                )

            date_df_temp3['Date'] = pd.to_datetime(date_df_temp3['Date'])

            date_formatted3 = date_df_temp3['Date'].dt.strftime('%Y-%m-%d')

            # Create a hover template to customize the tooltip content
            hover_template = (
                '<b>Date</b>: %{customdata[0]}<br>' +
                '<b>SODP</b>: %{customdata[1]:,.0f}<br>'
                '<extra></extra>'
            )

            # Prepare custom data by stacking the converted 'Date' column and other columns
            customdata = np.stack((date_formatted3, date_df_temp3['30days-prior']), axis=-1)

            # Update the hovertemplate for the figure and add custom data
            fig.update_traces(hovertemplate=hover_template, customdata=customdata)
            fig.update_layout(title_text=f'<b>Delta 30days-prior for flight {selected_3}</b>')
            fig.update_layout(width=800)
            st.plotly_chart(fig)
            print("")
               
elif side_bar == '📝Help: CM':
    col1, col2 = st.columns((5,5))       
    with col1:
        st.markdown(""" ## Classification Model Documentation""")
    with col2:
        st.image("https://media.giphy.com/media/H1dXomvQ0jxNLASIqK/giphy.gif", width = 200)
        
    st.markdown(""" 

### Overview

This code performs a flight capacity prediction task using historical data for different flights. The goal is to predict whether a flight will be full 30 days prior to its departure date. The code uses the `pandas`, `sqlalchemy`, `tqdm`, and `scikit-learn` libraries to read data from a SQL database, process it, build a predictive model using XGBoost, and evaluate the model's performance.

### Prerequisites

Before running the code, ensure that you have the following libraries installed:

- `pandas`: For data manipulation and analysis.
- `sqlalchemy`: For creating a connection to the SQL database.
- `tqdm`: For creating a progress bar to monitor the processing of chunks.
- `scikit-learn`: For building the predictive model and evaluating its performance.
- `xgboost`: For the XGBoost classifier.

### Steps

1. Import the necessary libraries:

```python
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score, accuracy_score
```

2. Connect to the SQL database using the `create_engine` function from `sqlalchemy`.

3. Define the SQL query to retrieve distinct flight numbers (`fltnum`) from the database, sorted in descending order.

4. Set the `chunksize` to determine the number of rows to be processed at a time. This is useful for handling large datasets efficiently.

5. Calculate the total number of rows and chunks to be processed.

6. Initialize a progress bar using `tqdm` to track the processing of chunks.

7. Loop through the chunks obtained from the SQL query and perform the following steps for each chunk:

    a. Process each flight number (`fltnum`) in the chunk:
    
        - Retrieve data for the specific flight from the database.
        - Preprocess the data, filling NaN values, and calculating a new feature `sodp`.
        - Check if the flight is always full, never full, or if it requires balancing.
        - If the flight requires balancing due to an imbalanced class distribution, perform oversampling using the `resample` function.
        - Encode categorical features using frequency encoding.
        - Split the data into training and testing sets based on the snapshot date.
        - Build an XGBoost classifier model and fit it to the training data.
        - Predict the labels for the test data and evaluate the model's performance using ROC AUC score and accuracy.

    b. Update the progress bar after processing each chunk.

8. Close the progress bar after processing all chunks.


### Note

Please ensure that you have configured the `engine` object to correctly connect to your SQL database before running the code. Additionally, ensure that all the required libraries are installed and accessible in your Python environment. The code makes use of `XGBoost`, so it's essential to have it installed as well.

This documentation serves as a high-level overview of the code and its functionality. If you have any questions or need further details, feel free to ask.""")

    st.markdown("""# Classifier Model Evaluation and Comparision

## Overview

I developed a flight capacity prediction model to forecast whether a flight will be fully booked 30 days prior to its departure date. Throughout the project, I experimented with several machine learning algorithms, including Random Forest Classifier, XGBoost Classifier, Gaussian Naive Bayes, Decision Tree Classifier, and Support Vector Machine (SVM). Each model was evaluated using relevant metrics, such as accuracy, ROC AUC score, and the confusion matrix.

## Data Processing

I started by preprocessing the data, including feature selection and target variable preparation. I extracted the features from the dataset and encoded the target variable to represent discrete classes.

## Model Evaluation

### Random Forest Classifier

I trained the Random Forest Classifier on the training data and predicted the labels for the testing data. The confusion matrix visualized the model's performance, showcasing the number of true positives, true negatives, false positives, and false negatives.

### Hyperparameter Tuning with GridSearchCV

To optimize the Random Forest Classifier, I performed hyperparameter tuning using GridSearchCV. By trying various combinations of hyperparameters (e.g., n_estimators, max_features, max_depth, and max_leaf_nodes), I found the best model configuration. After fitting the model with the best hyperparameters, and I re-evaluated its performance.

### Hyperparameter Tuning with RandomizedSearchCV

For a different approach to hyperparameter tuning, I used RandomizedSearchCV. This technique randomly selects hyperparameter values from a parameter grid, and I evaluated the model with these random hyperparameters. The best combination of hyperparameters was determined, and the model was refitted with these values. 

### XGBoost Classifier

Moving on to the XGBoost Classifier, I trained the model on the training data and predicted the labels for the testing data. 

### Gaussian Naive Bayes

For the Gaussian Naive Bayes model, I trained the classifier on the training data and predicted the labels for the testing data. 

### Decision Tree Classifier

Next, I evaluated the Decision Tree Classifier. 

### Support Vector Machine (SVM)

Finally, I evaluated the Support Vector Machine model with a linear kernel. The model was trained on the training data and predicted the labels for the testing data. 

## Conclusion

I tested various machine learning models for flight capacity prediction and evaluated their performances. The Random Forest Classifier with hyperparameter tuning using GridSearchCV demonstrated the highest accuracy and ROC AUC score, making it the most suitable model for this task. However, the final model selection should be based on specific project requirements and consideration of other factors, such as interpretability and computational resources. This evaluation process provided valuable insights into the strengths and weaknesses of each model, enabling me to make informed decisions for flight capacity prediction.""")   

elif side_bar == "Regression Model":  
     
    text,img2 = st.columns((2,1))
    with text:
        st.title("SODP Regression Model")
        st.markdown("Predicts exactly when the flight will get full")  
        st.markdown("Trained on historical data from `2017-2018`") 
        st.markdown("Predicts `2019` flight") 
        st.write("")
        st.write("")
        
    with img2:
        st.image("https://media.giphy.com/media/cNZQpCC8kY60gnnd0n/giphy.gif", width = 150)
    
    option_1col,option_2col = st.columns((5,5))
    
    with option_1col:
        df1 = pd.read_sql("SELECT DISTINCT lego FROM DEMODATA ORDER BY lego", get_connection())
        options_1 = df1['lego'].tolist()
        default_option_index1 = options_1.index('JFK')
        selected_option_1 = st.selectbox('Select Leg Origin', options_1, index=default_option_index1)

        df2 = pd.read_sql(f"SELECT DISTINCT legd FROM DEMODATA WHERE lego = '{selected_option_1}' ORDER BY legd", get_connection())
        options_2 = df2['legd'].tolist()
        # default_option_index2 = options_2.index('LHR')
        selected_option_2 = st.selectbox('Select Leg Destination', options_2)

    with option_2col:
        df3 = pd.read_sql(f"SELECT DISTINCT fltnum FROM DEMODATA WHERE lego = '{selected_option_1}' AND legd = '{selected_option_2}' ORDER BY fltnum", get_connection())
        options_3 = df3['fltnum'].tolist()
        # default_option_index3 = options_3.index('0001')
        selected_option_3 = st.selectbox('Select Flight Number', options_3 )

        slider_value = st.slider("Select Days prior:", min_value=0, max_value=341, value=170)
        
    df4 = pd.read_sql(f"SELECT DISTINCT fltdep FROM DEMODATA WHERE lego = '{selected_option_1}' AND legd = '{selected_option_2}' AND fltnum = '{selected_option_3}' AND fltdep > '2018-12-31' ORDER BY fltdep", get_connection())
    options_4 = df4['fltdep'].tolist()
    options_4_datetime = [datetime.strptime(option, "%Y-%m-%d") for option in options_4]
    selected_option_4 = st.date_input(f"Select dept date: ", options_4_datetime[0])
    selected = selected_option_4.strftime("%Y-%m-%d")
    if selected not in options_4:
        st.write("⚠️ No flight available for prediction!")
        
    else:
        st.write("You have selected:")
        st.write('Origin:', selected_option_1, ", Destination: ", selected_option_2, ", Flt num: ", selected_option_3, ", Days prior: ", slider_value)
        
        if st.button("PREDICT"):
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.markdown("### Chosen values")
            st.sidebar.markdown(f"OD: `{selected_option_1} - {selected_option_2}`\n")
            st.sidebar.markdown(f"Fltnum: `{selected_option_3}`\n")
            st.sidebar.markdown(f"DP: `{slider_value}`\n")
            st.sidebar.markdown(f"Dept Date: `{selected_option_4}`")
            
            with open(f"Fltnum{selected_option_3}_regression_model_{slider_value}_fullbk.pickle", "rb") as f:
                model = pickle.load(f)
            
            flight = fetch_flight_data(selected_option_3)
            flight['fltdep'] = pd.to_datetime(flight['fltdep'])
            flight['snap'] = pd.to_datetime(flight['snap'])
            flight['days_prior'] = (flight['fltdep'] - flight['snap']).dt.days
            flight['sodp'] = np.where(flight['bkd'] > flight['cap'], (flight['fltdep'] - flight['snap']).dt.days, -1)

            # Find the maximum 'sodp' value for each 'fltdep' group and broadcast it to the original DataFrame
            flight['max_sodp'] = flight.groupby('fltdep')['sodp'].transform('max')
            # Replace 'sodp' with 'max_sodp' for each 'fltdep' group
            flight['sodp'] = flight['max_sodp']
            flight.drop(columns=['max_sodp'], inplace=True)

            target_date = pd.to_datetime('2019-01-01')
            distinct_flights2 = flight[flight['fltdep'] >= target_date]
            max_sodp_flights2 = distinct_flights2[distinct_flights2['snap'] == distinct_flights2['fltdep']]
            final2 = max_sodp_flights2[['fltdep', 'sodp', 'frc', 'cap', 'rev','frc_unc']]

            distinct_flights2 = distinct_flights2[distinct_flights2['days_prior'] >= slider_value]

            # Calculate mean bkd for each 'fltdep' and 'days_prior' combination
            distinct_flights2['bkd_mean'] = distinct_flights2.groupby(['fltdep', 'days_prior'])['bkd'].transform('max')

            # Pivot the table to create separate columns for each 'days_prior' value
            pivoted_df2 = distinct_flights2.pivot_table(index='fltdep', columns='days_prior', values='bkd_mean', fill_value=0)

            # Add a prefix to the column names for clarity
            pivoted_df2.columns = ['bkd_' + str(col) for col in pivoted_df2.columns]

            # Reset the index to turn 'fltdep' back into a regular column
            pivoted_df2.reset_index(inplace=True)

            pivoted_df2['fltdep'] = pd.to_datetime(pivoted_df2['fltdep'])

            # Extracting day, month, and year
            pivoted_df2['day'] = pivoted_df2['fltdep'].dt.day
            pivoted_df2['month'] = pivoted_df2['fltdep'].dt.month
            pivoted_df2['year'] = pivoted_df2['fltdep'].dt.year

            # Extracting day of the week (1: Monday, 7: Sunday)
            pivoted_df2['day_of_week'] = pivoted_df2['fltdep'].dt.dayofweek + 1

            # Extracting week of the month
            pivoted_df2['week_of_month'] = pivoted_df2['fltdep'].dt.day.apply(lambda day: (day - 1) // 7 + 1)

            pivoted_df2.fillna(0, inplace=True)

            pivoted_df2.columns = pivoted_df2.columns.astype(str)
            final2['fltdep'] = pd.to_datetime(final2['fltdep'])
            date_df_temp2 = pivoted_df2.merge(final2[['fltdep', 'sodp', 'frc', 'cap', 'rev','frc_unc']], left_on='fltdep', right_on='fltdep', how='left')
            date_df_temp2.fillna(0, inplace=True)
            date_df_temp2['sodp'] = date_df_temp2['sodp'].astype(int)
            
            date_df_temp2['fltdep'] = pd.to_datetime(date_df_temp2['fltdep']).dt.date
            st.dataframe(date_df_temp2)
            
            feature_names = model.get_booster().feature_names
            
            feature_names = feature_names + ['sodp']
            date_df_temp2 = date_df_temp2[feature_names]
            
            filtered_df = date_df_temp2.drop('sodp', axis = 1)
            
            prediction = model.predict(filtered_df)
            actual = date_df_temp2['sodp']
            
            #PREDICTED
            start_date = '2019-01-01'
            end_date = '2019-12-31'
            full_date_range = pd.date_range(start=start_date, end=end_date)
            date_df = pd.DataFrame({'Date': full_date_range})
            date_df['Date'] = pd.to_datetime(date_df['Date'])

            date_df['sodp'] = prediction
            date_df_temp2 = date_df

            date_df_temp2['sodp'].fillna(2, inplace=True)
            date_df_temp2['sodp'] = date_df_temp2['sodp'].astype(int)

            date_df_temp2['Date'] = pd.to_datetime(date_df_temp2['Date'], format='%Y-%m-%d')

            fig = calplot(
                date_df_temp2, 
                x= "Date", 
                y = "sodp",
                dark_theme=False,
                years_title=True,
                gap=1,
                name="SODP",
                month_lines_width=2, 
                month_lines_color="black",
                colorscale="magma",
                showscale=True,
                cmap_max=max(prediction),
                cmap_min= min(prediction)
                )

            date_df_temp2['Date'] = pd.to_datetime(date_df_temp2['Date'])

            date_formatted2 = date_df_temp2['Date'].dt.strftime('%Y-%m-%d')

            hover_template = (
                '<b>Date</b>: %{customdata[0]}<br>' +
                '<b>SODP</b>: %{customdata[1]:,.0f}<br>'
                '<extra></extra>'
            )

            customdata = np.stack((date_formatted2, date_df_temp2['sodp']), axis=-1)

            fig.update_traces(hovertemplate=hover_template, customdata=customdata)
            fig.update_layout(title_text=f'<b>Predicted SODP for flight {selected_option_3}</b>')
            fig.update_layout(width=800)


            st.plotly_chart(fig)
            
            st.write("")
            #ACTUAL
            start_date = '2019-01-01'
            end_date = '2019-12-31'
            full_date_range = pd.date_range(start=start_date, end=end_date)
            date_df2 = pd.DataFrame({'Date': full_date_range})
            date_df2['Date'] = pd.to_datetime(date_df2['Date'])

            # Step 5: Combine the current testing data with the date_df
            date_df2['sodp'] = actual
            date_df_temp = date_df2

            # Step 6: Fill missing values in 'Value' with 2 
            date_df_temp['sodp'].fillna(2, inplace=True)
            date_df_temp['sodp'] = date_df_temp['sodp'].astype(int)

            # Add a Date column to the date_df_temp DataFrame
            date_df_temp['Date'] = pd.to_datetime(date_df_temp['Date'], format='%Y-%m-%d')

            fig = calplot(
                date_df_temp, 
                x= "Date", 
                y = "sodp",
                dark_theme=False,
                years_title=True,
                gap=1,
                name="SODP",
                month_lines_width=2, 
                month_lines_color="black",
                colorscale="magma",
                showscale=True,
                cmap_max=max(prediction),
                cmap_min= min(prediction)
                )

            date_df_temp['Date'] = pd.to_datetime(date_df_temp['Date'])

            date_formatted = date_df_temp['Date'].dt.strftime('%Y-%m-%d')

            # Create a hover template to customize the tooltip content
            hover_template = (
                '<b>Date</b>: %{customdata[0]}<br>' +
                '<b>SODP</b>: %{customdata[1]:,.0f}<br>'
                '<extra></extra>'
            )

            # Prepare custom data by stacking the converted 'Date' column and other columns
            customdata = np.stack((date_formatted, date_df_temp['sodp']), axis=-1)

            # Update the hovertemplate for the figure and add custom data
            fig.update_traces(hovertemplate=hover_template, customdata=customdata)
            fig.update_layout(title_text=f'<b>Actual SODP for flight {selected_option_3}</b>')
            fig.update_layout(width=800)
            st.plotly_chart(fig)
            
            st.write("")
            
            #Difference
            start_date = '2019-01-01'
            end_date = '2019-12-31'
            full_date_range = pd.date_range(start=start_date, end=end_date)
            date_df3 = pd.DataFrame({'Date': full_date_range})
            date_df3['Date'] = pd.to_datetime(date_df3['Date'])

            # Step 5: Combine the current testing data with the date_df
            date_df3['sodp'] = date_df2['sodp'] - date_df['sodp']
            date_df_temp3 = date_df3

            # Step 6: Fill missing values in 'Value' with 2 
            date_df_temp3['sodp'].fillna(2, inplace=True)
            date_df_temp3['sodp'] = date_df_temp3['sodp'].astype(int)
            max_value = date_df_temp3['sodp'].max()

            # Add a Date column to the date_df_temp DataFrame
            date_df_temp3['Date'] = pd.to_datetime(date_df_temp3['Date'], format='%Y-%m-%d')

            fig = calplot(
                date_df_temp3, 
                x= "Date", 
                y = "sodp",
                dark_theme=False,
                years_title=True,
                gap=1,
                name="SODP",
                month_lines_width=0.5, 
                month_lines_color="black",
                colorscale="RdBu",
                showscale=True,
                cmap_max = max_value, 
                cmap_min = - max_value
                )

            date_df_temp3['Date'] = pd.to_datetime(date_df_temp3['Date'])

            date_formatted3 = date_df_temp3['Date'].dt.strftime('%Y-%m-%d')

            # Create a hover template to customize the tooltip content
            hover_template = (
                '<b>Date</b>: %{customdata[0]}<br>' +
                '<b>SODP</b>: %{customdata[1]:,.0f}<br>'
                '<extra></extra>'
            )

            # Prepare custom data by stacking the converted 'Date' column and other columns
            customdata = np.stack((date_formatted3, date_df_temp3['sodp']), axis=-1)

            # Update the hovertemplate for the figure and add custom data
            fig.update_traces(hovertemplate=hover_template, customdata=customdata)
            fig.update_layout(title_text=f'<b>Delta SODP for flight {selected_option_3}</b>')
            fig.update_layout(width=800)
            st.plotly_chart(fig)
        
            absolute_error = (prediction - actual)

            start_date = '2019-01-01'
            end_date = '2019-12-31'
            date_range = pd.date_range(start=start_date, end=end_date)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=date_range, y=absolute_error, mode='markers', name='Absolute Error'))
            fig.add_trace(go.Scatter(x=date_range, y=np.zeros(len(absolute_error)), mode='lines', line=dict(color='red', dash='dash'), name='Zero Error'))

            fig.update_layout(title='Error Between Predicted and Actual Values',
                            xaxis_title='Departure Dates',
                            yaxis_title='Actual Error',
                            showlegend=True)

            st.plotly_chart(fig)
            
            st.write("Depiction of holidays")
            data = {
                        'Holiday': ['New Year\'s Day', 'St. Patrick\'s Day', 'Easter Sunday', 'Holi', 'Ramadan', 'Memorial Day',
                                    'Eid al-Fitr', 'Summer Break', 'Eid al-Adha', 'Labor Day', 'Islamic New Year', 'Diwali', 'Halloween',
                                    'Thanksgiving Day', 'Christmas Day', 'Chinese New Year'],
                        '2017 Dates': ['January 1', 'March 17', 'April 16', 'March 13', 'May 27 - June 24', 'May 29', 'June 25', 'June - August',
                                    'September 1', 'September 4', 'September 21', 'October 19', 'October 31', 'November 23', 'December 25', 'January 28'],
                        '2018 Dates': ['January 1', 'March 17', 'April 1', 'March 2', 'May 16 - June 14', 'May 28', 'June 15', 'June - August',
                                    'August 21', 'September 3', 'September 11', 'November 7', 'October 31', 'November 22', 'December 25', 'February 16'],
                        '2019 Dates': ['January 1', 'March 17', 'April 21', 'March 21', 'May 6 - June 3', 'May 27', 'June 4', 'June - August',
                                    'August 11', 'September 2', 'August 31', 'October 27', 'October 31', 'November 28', 'December 25', 'February 5'],
                    }
            df = pd.DataFrame(data)
            st.table(df)

            #BOOKING CURVE DEPENDING ON CHOSEN DEPTDATE
            s = pd.to_datetime(selected_option_4)
            date_only = s.date()
            date_string = date_only.strftime("%Y-%m-%d")    
            
            particular_date_datetime = pd.to_datetime(selected_option_4)

            req_df = pd.read_sql(f"SELECT snap, bkd, cap FROM DEMODATA WHERE lego = '{selected_option_1}' AND legd = '{selected_option_2}' AND fltnum = '{selected_option_3}' AND fltdep = '{selected_option_4}' AND cmp = 'Y' ORDER BY snap", get_connection())
            req_df['snap'] = pd.to_datetime(req_df['snap'])
            selected_option_4 = pd.to_datetime(selected_option_4)
            req_df['days_prior'] = (selected_option_4 - req_df['snap']).dt.days
            req_df = req_df.sort_values(by='days_prior', ascending=True)
            used = req_df[req_df['days_prior'] >= slider_value]
            available = req_df[req_df['days_prior'] < slider_value]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=used['days_prior'], y=used['bkd'], mode='lines', name=f'Booked as of {slider_value} DP', visible= True))
            fig.add_trace(go.Scatter(x=available['days_prior'], y=available['bkd'], mode='lines', name='Booked from entire available data', line=dict(color='green', dash='dot'), visible= True))
            fig.add_trace(go.Scatter(x=req_df['days_prior'], y=req_df['cap'], mode='lines', name='Capacity', visible= True))
            
            
            #predicted
            pred_for_select_dept = date_df_temp2[date_df_temp2['Date'] == particular_date_datetime]
            days_pred1 = np.int64(pred_for_select_dept['sodp'])
            days_pred1 = int(days_pred1)

            # new_date_datetime1 = particular_date_datetime - timedelta(days = days_pred1)
            # new_date1 = new_date_datetime1.strftime(format='%Y-%m-%d')

            if days_pred1 != -1:
                fig.add_trace(go.Scatter(x=[days_pred1, days_pred1], y=[500, 1], mode='lines', line=dict(color='red', width=1), name='Predicted SODP'))

            #actual
            actual_for_select_dept = date_df_temp[date_df_temp['Date'] == particular_date_datetime]
            days_act1 = np.int64(actual_for_select_dept['sodp'])
            days_act1 = int(days_act1)

            # date_datetime1 = particular_date_datetime - timedelta(days = days_act1)
            # date2 = date_datetime1.strftime(format='%Y-%m-%d')
            
            if days_act1 != -1:
                fig.add_trace(go.Scatter(x=[days_act1, days_act1], y=[500, 1], mode='lines', line=dict(color='yellow', width=1), name='Actual SODP'))

            fig.update_layout(title=f'Plotting for dept date: {date_string}',
                            xaxis_title='Days Prior',
                            yaxis_title='PAX',
                            showlegend=True)
            fig.update_layout(width=800)
            fig.update_layout(xaxis=dict(autorange='reversed'))
            # fig.update_xaxes(tickformat='%Y-%m-%d')
            st.plotly_chart(fig)
            if days_pred1 == -1:
                st.markdown(f"Predicted SODP: `Flight doesn't sell out`")
            else:
                st.markdown(f"Predicted SODP: `{days_pred1}`")
        
            st.write("Steep increases and decreases show group bookings or cancellations.")
            st.write(f"""The dotted line depicts how the actual booking curve is. It is extracted from the given 2019 dataset.\n The line in blue depicts the booking curve {slider_value} days prior to departure.""")
                
elif side_bar == "📝Help: RM":
    col1, col2 = st.columns((5,5))
    
    with col1:
        st.markdown(""" ## Regression Model Documentation""")
    with col2:
        st.image("https://media.giphy.com/media/VbnLt7V0geYuPWObo0/giphy.gif", width = 200)
    
    st.markdown("""

This documentation provides an overview of the process and steps involved in training an XGBoost regression model for flight booking prediction using historical flight data. The code uses Python and several libraries, including NumPy, pandas, tqdm, scikit-learn, xgboost, and concurrent.futures. The purpose of the code is to train models for different days prior to a flight's departure and save the models to pickle files.

## 1. Importing Libraries

The first step is to import the necessary Python libraries required for the model training process:

```python
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import concurrent.futures
```

## 2. Data Preparation and Model Training

### 2.1. Function Definition

Next, a function `train_model(count)` is defined to train an XGBoost regression model for a specified number of days prior to the flight departure:

```python
def train_model(count):
    # ... (implementation details of the function)
```

### 2.2. Data Preprocessing

The function performs data preprocessing on the training and testing datasets, which includes reading data from a SQL database and applying necessary transformations:

- **Training Data**: Data is read from the `FLT_TRAIN_DATA` table for a specific flight number and a maximum number of days prior to departure (`count`). Features are generated for each flight date and days prior to departure, and the data is filtered to keep only the records corresponding to the maximum days prior to departure. Irrelevant columns are dropped, and the dataset is prepared for training.

- **Testing Data**: Data is read from the `FLT_TEST_DATA` table for the same flight number, but this time, it considers only the flights with departure dates after 2018-12-31. Similar feature generation and data preparation steps are applied to the testing dataset.

### 2.3. Principal Component Analysis (PCA)

The function then performs Principal Component Analysis (PCA) on the training features. The data is standardized using `StandardScaler`, and PCA is applied to reduce the dimensionality to 3 principal components:

```python
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)

model = PCA(n_components=3).fit(scaled_train_features)
X_pc = model.transform(scaled_train_features)
```

### 2.4. XGBoost Model Training

The XGBoost regression model is trained using the `XGBRegressor` from the `xgboost` library. The most important features are selected based on the PCA results, and the model is trained with the best hyperparameters:

```python
xgb_r = xg.XGBRegressor(objective='reg:squarederror', seed=123, **best_params)
xgb_r.fit(X_train2, y_train2)
```

### 2.5. Model Evaluation and Saving

The trained model is evaluated using Mean Absolute Error (MAE) on the testing data. The MAE value is printed for each `count`. The trained models are stored in a dictionary `models` and later saved to separate pickle files for each `count`.

## 3. Multi-threading for Model Training

The model training process is multi-threaded using `concurrent.futures.ThreadPoolExecutor`. It allows the function `train_model` to be executed concurrently for different `counts`, making the training process faster.

```python
# ... (code to create ThreadPoolExecutor and train models concurrently)
```

## 4. Saving Trained Models

After the model training is completed, each trained model in the `models` dictionary is saved as a pickle file with a filename based on the `count` value and flight number:

```python
for counter, model in models.items():
    with open(f'Fltnum0784_model_{counter}_fullbk.pickle', 'wb') as f:
        pickle.dump(model, f)
```

## 5. Conclusion

The code provided demonstrates how to train XGBoost regression models using Principal Component Analysis on historical flight data for a specific flight number. The models are trained for different days prior to flight departure, and the trained models are saved for future use. The multi-threading technique is utilized to speed up the training process for different days prior to the flight.")
""") 
    
    
    
    

    
    
    
    

    





# Disaster Response Pipeline Project
This project is developed as a final assignment for Data engineering course as a part of Data Science nanodegree in
Udacity

Project consist of 3 parts:

1. ETL Pipeline Preparation
    - Preprocessing of Raw data and Saving to Database
2. ML Pipeline Preparation
    - Trainin Multioutput classification algorithm
    - After iterating Logistic regression, Random Forest and XgBoost with different hyperparameters the best algorithm was chose
   
    <img width="568" alt="Screenshot 2022-08-28 at 22 56 21" src="https://user-images.githubusercontent.com/86057193/187096550-50d9448a-0616-405a-9fd8-a336752d8549.png">


3. Web App

### Instructions how to run the app:
0. Before running commands below change working directory in Terminal using command cd

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

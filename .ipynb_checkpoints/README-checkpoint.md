# Disaster Response Pipeline Project
This project is developed as a final assignment for Data engineering course as a part of Data Science nanodegree in
Udacity

Project consist of 3 parts:

1. ETL Pipeline Preparation
    - Preprocessing of Raw data and Saving to Database
2. ML Pipeline Preparation
    - Trainin Multioutput classification algorithm
    - After iterating Logistic regression, Random Forest and XgBoost with different hyperparameters the best algorithm was chose
   
   

# Project Motivation
The goal of the project is to apply data preprocessing and machine learning pipelines in User intuitive Web App.
Ml algorithm is trained on messages that has multiple characteristics descirbed in categories file. 

The full scale deployment of this app that scans appearing texts in media, social networks, and blogs around the globe can play a role of early warning system that detects disasters on early stages and give humanity an opportunity to prevent negative consequences

3. Web App

<<<<<<< HEAD
### File Descriptions

* app
    | - templates
    | |- master.html # main page of web app
    | |- go.html # classification result page of web app
    |- run.py # Flask file that runs app

* data
    |- disaster_categories.csv # input data
    |- disaster_messages.csv # input data
    |- process_data.py # data processing code
    |- DisasterResponse.db # database to save processed data

* models
    |- train_classifier.py # machine learning code
    |- classifier.pkl # output model

- README.md

* notebooks # folder not used in the app, a.k.a sanbox where analytics drafts are collected
    |- ETL Pipeline Preparation.ipynb # ETL notebook
    |- ML Pipeline Preparation.ipynb # ML notebook
    |- categories.csv # input data
    |- messages.csv # input data
    |- project_database.db # output data



=======

### Instructions:

0. Change a current directory to the working folder using cd command

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python ~/app/run.py`

3. Go to http://0.0.0.0:3001/


<img width="1440" alt="Screenshot 2022-08-28 at 22 19 55" src="https://user-images.githubusercontent.com/86057193/187096612-4b15be70-5203-4036-b05d-f6a9b9020b2b.png">

<img width="1440" alt="Screenshot 2022-08-28 at 22 20 08" src="https://user-images.githubusercontent.com/86057193/187096617-13b291a4-12d7-43c9-a0d4-0ada68be7db6.png">

<img width="1440" alt="Screenshot 2022-08-28 at 22 20 14" src="https://user-images.githubusercontent.com/86057193/187096621-0f3ddb9b-eb72-4bd0-bbf6-67a36c58d99f.png">


 <img width="568" alt="Screenshot 2022-08-28 at 22 56 21" src="https://user-images.githubusercontent.com/86057193/187096550-50d9448a-0616-405a-9fd8-a336752d8549.png">


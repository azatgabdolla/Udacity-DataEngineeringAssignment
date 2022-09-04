# Disaster Response Pipeline Project

This project is developed as a final assignment for Data engineering course as a part of Data Science nanodegree in
Udacity


# Project Motivation
The goal of the project is to apply data preprocessing and machine learning pipelines in User intuitive Web App.
Ml algorithm is trained on messages that has multiple characteristics descirbed in categories file. 

The full scale deployment of this app that scans appearing texts in media, social networks, and blogs around the globe can play a role of early warning system that detects disasters on early stages and give humanity an opportunity to prevent negative consequences



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

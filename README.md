# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to the app's directory and run the following commands to start your web app.
    ```sh
    $ env | grep WORK
    ```
3. Copy **SPACEDOMAIN** and **SPACEID** for next step.
4. Build URL [http://{SPACEID}-3001.{SPACEDOMAIN}]
    ```sh
    $ python run.py
    ```

5. Go to the URL build in STEP 4

# Import credentials
from dotenv import load_dotenv
load_dotenv()

# Import libraries
import pandas as pd
import numpy as np
import argparse
import time
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":

    # MLflow tracking setup
    EXPERIMENT_NAME="fraud_detector_RFC"
    mlflow.set_tracking_uri("https://cpellerin-mlflow-christophe-fraud.hf.space")
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    # Training
    print("🏃 Start train...")
    start_time = time.time()
    
    ## Call MLFlow autolog
    ## We won't log models right away
    mlflow.sklearn.autolog(log_models=False) 

    ##############################################################
    ## HYPERPARAMETERS

    ## Parse arguments given in shell script
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", default=5)
    parser.add_argument("--min_samples_split", default=10)
    args = parser.parse_args()
  
    ##############################################################


    # Import data
    print("🏃 Loading dataset...")
    data = pd.read_csv("C:/Users/Jedha1/Desktop/Bloc_4_Fraud_Detection_Plus/data/fraudTest.csv", index_col=0)
    data = data.astype({col: "float64" for col in data.select_dtypes(include=["int"]).columns})

    # Imbalanced dataset!!!
    count_class_0, count_class_1 = data['is_fraud'].value_counts()

    data_class_0 = data[data['is_fraud'] == 0]
    data_class_1 = data[data['is_fraud'] == 1]

    # Reduce Class 0 to Class 1 level
    data_class_0_under = data_class_0.sample(55000, random_state=42)
    #data_class_0_under = data_class_0.sample(count_class_1, random_state=42)

    # Concatenate new Class 0 and Class 1
    data_balanced = pd.concat([data_class_0_under, data_class_1])

    # Shuffle data to get a random order
    data = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate target variable y from features X
    print("🏃 Separating labels from features...")
    X = data.drop("is_fraud", axis=1)
    y = data["is_fraud"]

    # Divide data Train set & Test set with test_size = 0.2 (20%)
    print("🏃 Dividing into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)

    ## Features engineering
    def features_engineering(data):
      
        # Create new features
        data = data.copy()
        
        # Extract date information
        data['weekday'] = pd.to_datetime(data['trans_date_trans_time']).dt.day_name()
        data['hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour

        # Enrichment with lenght of credit card number
        data['cc_num'] = data['cc_num'].astype(str)
        data['lenght_cc_num'] = data['cc_num'].str.len()
  
        # Convert date of birthday in age (from today)
        data['age'] = pd.to_datetime('today').date().year - pd.to_datetime(data['dob']).dt.year

        # Enrichment with distance (kms) between customer and merchant
        def haversine(long, merch_long, lat, merch_lat):
            # Convert degrees to radians
            long, merch_long, lat, merch_lat = map(np.radians, [long, merch_long, lat, merch_lat])
            diff_lon = merch_long - long
            diff_lat = merch_lat - lat
            # earth radius: 6371 kms
            distance_km = 2*6371*np.arcsin(np.sqrt(np.sin(diff_lat/2.0)**2 + np.cos(lat) * np.cos(merch_lat) \
                                                * np.sin(diff_lon/2.0)**2))
            return distance_km
        data.loc[:, 'distance_km'] = data.apply(lambda x: haversine(x['long'], x['merch_long'], x['lat'], x['merch_lat']), axis = 1)
       
        # Anonymize (because of compliance) and drop useless data (or too many categories)
        useless_columns= ['trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'gender', 'street',
                           'city', 'zip', 'state', 'lat', 'long', 'job', 'dob', 'trans_num', 'unix_time',
                           'merch_lat', 'merch_long']
        
        data = data.drop(useless_columns, axis=1)
        data = data.astype({col: "float64" for col in data.select_dtypes(include=["int"]).columns})

        return data

    engineering_preprocessor = FunctionTransformer(features_engineering)

    # Preprocessing 
    print("🏃 Preprocessing pipeline...")
    X_train_after_engineering = features_engineering(X_train)
    
    ## Categorical features X_train
    ## Select all the columns containing strings
    categorical_features = X_train_after_engineering.select_dtypes("object").columns
    ## No missing values, only OHE
    categorical_transformer = OneHotEncoder(drop='first')

    ## Numeric features X_train
    ## Select all the columns containing anything else than strings
    numeric_feature_mask = ~X_train_after_engineering.columns.isin(X_train_after_engineering.select_dtypes("object").columns)
    numeric_features = X_train_after_engineering.columns[numeric_feature_mask]
    numeric_transformer = StandardScaler()

    transforming_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical_transformer", categorical_transformer, categorical_features),
            ("numeric_transformer", numeric_transformer, numeric_features)
        ]
    )

    # Pipeline 
    n_estimators = int(args.n_estimators)
    min_samples_split=int(args.min_samples_split)


    ## BASELINE : RANDOM FOREST CLASSIFIER
    print("🎯 Machine Learning - Model Baseline : RANDOM FOREST CLASSIFIER")
    model = Pipeline(steps=[
        ("Features_engineering", engineering_preprocessor),
        ('Features_transforming', transforming_preprocessor),
        ("Classifier",RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=42))
    ])
    
    # Log experiment to MLFlow
    with mlflow.start_run(experiment_id = experiment.experiment_id) as run:
        print("🏃 Log model to MLFlow...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)
        print("✅ Model trained!")
        print(f"---Total training time: {time.time()-start_time:.2f} seconds")

        # Log model seperately to have more flexibility on setup 
        # Record in registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=EXPERIMENT_NAME,
            registered_model_name="fraud_detector_Christophe_RFC_",
            signature=infer_signature(X_train, predictions))
        print("✅ Model logged in MLflow")

        # Get last model's version
        client = MlflowClient()
        latest = client.get_latest_versions(
            "fraud_detector_Christophe_RFC_", stages=["None"]
        )
        if latest:
            model_version = latest[-1].version
            print(f"[INFO] Model logged as version {model_version}")

            # Update alias "candidate"
            client.set_registered_model_alias(
                name="fraud_detector_Christophe_RFC_",
                alias="production",
                version=model_version,
            )
            print(f"[INFO] Alias 'production' now points to version {model_version}")
        else:
            print("[WARN] Aucun modèle trouvé dans le registre.")

        

        
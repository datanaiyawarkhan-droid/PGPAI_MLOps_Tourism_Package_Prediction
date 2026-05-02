# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/datanai-yawarkhan/Tourism-Package-Prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# Drop ID column which has no relevance for prediction
tourism_dataset.drop(columns=['CustomerID'], inplace=True)

# Encoding the categorical columns
label_encoder = LabelEncoder()
tourism_dataset['TypeofContact'] = label_encoder.fit_transform(tourism_dataset['TypeofContact'])
tourism_dataset['Occupation'] = label_encoder.fit_transform(tourism_dataset['Occupation'])
tourism_dataset['Gender'] = label_encoder.fit_transform(tourism_dataset['Gender'])
tourism_dataset['MaritalStatus'] = label_encoder.fit_transform(tourism_dataset['MaritalStatus'])
tourism_dataset['Designation'] = label_encoder.fit_transform(tourism_dataset['Designation'])
tourism_dataset['ProductPitched'] = label_encoder.fit_transform(tourism_dataset['ProductPitched'])

# List of numerical features in the dataset
numeric_features = [
    'Age',               # Customer's age
    'CityTier',          # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'NumberOfPersonVisiting',        # Number of people accompanying the customer on the tour
    'PreferredPropertyStar',     # Preferred hotel rating by customer
    'NumberOfTrips',         # Average number of trips customer takes annually
    'Passport',    # Whether the customer holds an active passport (binary: 0:No or 1:Yes)
    'OwnCar',    # Whether the customer owns a car (binary: 0:No or 1:Yes)
    'NumberOfChildrenVisiting',    # number of children visiting along with customer
    'MonthlyIncome',    # Monthly income of customer
    'PitchSatisfactionScore',    #Score indicating the customer's satisfaction with the sales pitch
    'NumberOfFollowups',    #Total number of follow-ups by the salesperson after the sales pitch
    'DurationOfPitch'    #Duration of the sales pitch delivered to the customer  
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # The method by which the customer was contacted (Company Invited or Self Inquiry)
    "Occupation",    # Customer's occupation (e.g., Salaried, Freelancer)
    "Gender",    # Gender of the customer (Male, Female)
    "MaritalStatus",    # Marital status of the customer (Single, Married, Divorced)
    "Designation",    # Customer's designation in their current organization
    "ProductPitched"    # The type of product pitched to the customer
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="datanai-yawarkhan/Tourism-Package-Prediction",
        repo_type="dataset",
    )

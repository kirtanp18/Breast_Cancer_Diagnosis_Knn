import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Preprocess_data:
    def load_preprocess_data(self):
        
        # 1. Load the dataset
        # Read the CSV file. If there are any '?' characters, treat them as missing blank spaces.
        df = pd.read_csv('Data/breast_cancer.csv', header=None, na_values='?')
        
        # 2. Name the columns
        # The raw data doesn't have headers, so we give them human-readable names based on the dataset documentation.
        df.columns = ['Sample_Code_Number', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
        
        # 3. Handle missing data
        # If any patient is missing a measurement (which we marked as blank earlier), fill it with a 0.
        df = df.fillna(0)
       
        # 4. Separate the Features (X) from the Answer (y)
        # 'X' holds the medical measurements. We drop the Patient ID (irrelevant) and the Class (the answer).
        X = df.drop(["Sample_Code_Number", "Class"], axis=1).values
        
        # 'y' holds the actual diagnosis (Class 2 for Benign, 4 for Malignant)
        y = df["Class"].values
        
        # 5. Scale the data
        # Some measurements might range from 1-10, others from 100-1000. 
        # MinMaxScaler shrinks all numbers proportionally so they fall exactly between 0 and 1.
        scale = MinMaxScaler()
        X = scale.fit_transform(X)
        
        # 6. Split into Training and Testing sets
        # test_size=0.3 means we save 30% of the data for testing, and use 70% to teach the model.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        # 7. Print the dimensions
        # This acts as a sanity check so we can see how many rows and columns our final sets have.
        print("Shape of X (All Features):", X.shape)
        print("Shape of y (All Answers):", y.shape)
        print("Shape of X_train (Training Features):", X_train.shape)
        print("Shape of X_test (Testing Features):", X_test.shape)
        print("Shape of y_train (Training Answers):", y_train.shape)
        print("Shape of y_test (Testing Answers):", y_test.shape)
        
        return X_train, X_test, y_train, y_test
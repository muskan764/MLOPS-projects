import pandas as pd
from sklearn.datasets import load_iris
import os

# Load the dataset
iris = load_iris()

# Create a DataFrame
df = pd.concat([pd.DataFrame(iris.data, columns=iris.feature_names), 
                pd.Series(iris.target, name="target")], axis=1)

# Create the 'data' folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save the CSV in the 'data' folder
df.to_csv("data/iris_dataset.csv", index=False)
print("Dataset saved successfully in the 'data' folder.")
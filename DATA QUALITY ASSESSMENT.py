import pandas as pd

# Load the dataset
file_path = r"C:\Users\raghu\Desktop\AIT 614\AIT 614 FINAL PROJECT\AIT 614 PROJECT PROPOSAL\cleaned_diabetes_dataset.csv"
df = pd.read_csv(file_path)

# Checking the Data Condition

# 1. Completeness
# Count missing values in each column
missing_values = df.isnull().sum()
# Calculate percentage of missing data
total_cells = df.shape[0] * df.shape[1]
percentage_missing = (missing_values.sum() / total_cells) * 100

print("Missing Values per Column:")
print(missing_values)
print("\nPercentage of Missing Data: {:.2f}%".format(percentage_missing))

# 2. Accuracy
expected_range = [0, 100]  
# Check accuracy of BMI values
incorrect_bmi = df[(df['BMI'] < expected_range[0]) | (df['BMI'] > expected_range[1])]

if len(incorrect_bmi) == 0:
    print("BMI values are within the expected range.")
else:
    print("There are some incorrect BMI values.")

# 3. Relevance
# Analyze the distribution of the 'HighBP' variable
high_bp_distribution = df['HighBP'].value_counts()

print("Distribution of HighBP variable:")
print(high_bp_distribution)


# 4. Validity
invalid_phys_activity = df[~df['PhysActivity'].isin([0, 1])]

if len(invalid_phys_activity) == 0:
    print("PhysActivity values are valid.")
else:
    print("Invalid PhysActivity values found.")




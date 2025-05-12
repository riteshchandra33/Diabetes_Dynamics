# Diabetes_Dynamics - A Data Analytics approach fopr uncovering trends and guiding health innovations

Overview

This project is dedicated to the analysis and prediction of diabetes risk factors utilizing data sourced from the file "diabetes_health_indicators.csv". Leveraging Databricks as the primary platform for data processing and analysis, the project encompasses several critical stages, including data cleaning, exploratory data analysis (EDA), addressing research questions, and implementing machine learning models.

Dataset

	•	Dataset Name: Diabetes Health Indicators
	•	Dataset Link: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

The dataset is structured within various folders, each containing pertinent files:

	•	DATABRICKS_PYTHON_FILES: Includes Python scripts for data processing and analysis.
	•	DATASET_INFO: Comprises the primary dataset file "Diabetes_health_indicators.csv" and its cleaned counterpart "Cleaned_diabetes_dataset.csv".
	•	DATABRICKS_HTML_FILES: Contains HTML files providing previews of code and outputs from Databricks notebooks.
	•	PYTHON_CODE: Houses Python scripts for data quality assessment and machine learning algorithms.

Data Cleaning

Databricks serves as the cornerstone of this project's data processing endeavors. To execute the provided code within your Databricks account, follow these sequential steps:

1. Initiate a cluster.
2. Open a new notebook.
3. Navigate to Files and opt for "Upload to DBFS".
4. Select "diabetes_health_indicators.csv" for upload.
5. Proceed with the directives outlined in the HTML file.

The dataset "diabetes_health_indicators.csv" underwent meticulous cleaning processes, encompassing tasks such as handling missing values and ensuring data integrity. Detailed documentation of the cleaning procedures can be found within the file "DATA_CLEANING.ipynb". Upon execution of the code, you will obtain the updated cleaned file "cleaned_diabetes_dataset.csv".

Following the acquisition of the cleaned dataset, proceed to upload the requisite files into Databricks to obtain the respective outputs.

Exploratory Data Analysis (EDA)

EDA was conducted employing the file "cleaned_diabetes_dataset.csv" to glean insights into the dataset's characteristics and identify pertinent patterns concerning diabetes risk factors. The code and resulting visualizations are accessible via the file "EDA.ipynb".

Research Questions Analysis

Subsequent to the EDA phase, the project delved into addressing specific research questions pertaining to diabetes risk factors. The code for these analyses is available in the file "ML_MODEL_ANALYSIS.ipynb". The questions encompass:

1. Predictive Power of Health Indicators and Lifestyle Factors
2. Influence of Socio-demographic Factors
3. Associations Between Behavioral Factors and Diabetes Risk

Jupyter Notebook

The supplementary aspect of the project was conducted and executed within a Jupyter Notebook environment, fostering interactive exploration and visualization of the analysis outcomes. Note: Ensure to upload the following files utilizing your respective local path.

Data Quality Assessment

A dedicated file named "AIT614_TEAM6_DATA QUALITY ASSESSMENT.py" houses the code for evaluating data quality independently, thereby assessing the dataset's integrity and consistency.

Machine Learning Models

The project entailed the development of machine learning models geared towards predicting diabetes risk based on identified factors. A diverse array of algorithms was explored, with their performance evaluated utilizing appropriate metrics. The requisite code and model implementations are provided in the file "AIT614_TEAM6_MODELS & ALGORITHMS.py".

Conclusion

This project epitomizes the application of data analysis and machine learning methodologies in scrutinizing diabetes risk factors. The insights garnered serve to augment our comprehension of the intricate interplay between health, socio-demographic, and behavioral determinants vis-à-vis diabetes prevalence.


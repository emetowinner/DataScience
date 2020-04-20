# Importing the pandas library into our work environment
import pandas as pd

# Using the pandas Excel class to raed the excel file load the excel file
data_set = pd.ExcelFile('ASSIGNMENT2_DATA.xlsx')
sheet = 0  # Sheet counter

"""Using the sheet name to pick each sheet from the file and read it to a 
    pandas dataframe using the read_excel function and then save to csv 
        file dynamically using the to_csv function """

for sheet_name in data_set.sheet_names:
    if sheet <= len(data_set.sheet_names):
        print(f'Reading {data_set.sheet_names[sheet]} to a DataFrame....')
        df = pd.read_excel(data_set, sheet_name)
        name = f'sheet{sheet}.csv'
        df.to_csv(name)
        print()
        print(
            f'Saved {data_set.sheet_names[sheet]} as a CSV file with the name {name}')
        sheet += 1


pandas
numpy
sklearn.model_selection
train_test_split
sklearn.ensemble
RandomForestClassifier
sklearn.preprocessing
LabelEncoder
sklearn.metrics
accuracy_score
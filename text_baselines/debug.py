import pandas as pd

df = pd.read_csv("data/project_1_3_data/IID/ADNI_binary_training.csv")
print(df.iloc[0])
print(df.iloc[0]['Generated_Text'])

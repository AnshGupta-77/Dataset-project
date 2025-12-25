import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler,OrdinalEncoder,MinMaxScaler,MaxAbsScaler,Normalizer
from category_encoders import BinaryEncoder, CountEncoder, TargetEncoder

df = pd.read_csv(r"C:\Users\DELL\Desktop\Self\Python\Dataset project\IndianFlightdata.csv")


print(df.shape())
print(df.info())

#Handling missing values

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df(col).mode()[0], inplace=True) 

#Fixing data types
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')

#removing duplicates
df.drop_duplicates(inplace=True)

#handling outliers

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1

    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

#removing irrelevant columns
drop_cols = [col for col in df.columns if 'id' in col.lower()]
df.drop(columns=drop_cols, inplace=True, errors='ignore')


#label encoding
df_label = df.copy()
le =  labelEncoder()
for col in cat_cols:
    if col in df_label.columns:
        df_label[col] = le.fit_transform(df_label[col])

#One-Hot encoding

df_onehot = pd.get_dummies(df, columns=cat_cols, drop_first=True)

#ordinal Encoding 

ordinal_cols = ['priority', 'performane']
ordinal_categories = [
    ['low', 'medium', 'high'],
    ['poor', 'average', 'good','excellent']
]

df_ordinal = df.copy()
if set(ordinal_cols).issubset(df.columns):
    ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
    df_ordinal[ordinal_cols] = ordinal_encoder.fit_transform(
        df_ordinal[ordinal_cols]
    )


#Binary Encoding
binary_encoder = CountEncoder(cols=cat_cols)
df_binary = binary_encoder.fit_transform(df)

#count Encoding
count_encoder = CountEncoder(cols-cat_cols)
df_count = count_encoder.fit_transform(df)

#targetEncoding

target_col = 'target'
df_target = df.copy()
if target_col in df.columns:
    target_encoder = TargetEncoder(cols-cat_cols)
    df_target = target_encoder.fit_transform(df, df[target_col])

#Feature Scaling

scalers = {
    "minmax": MinMaxScaler(),
    "maxabs": MaxAbsScaler(),
    "vector": Normalizer(norm="12"),
    "zscore": StandardScaler()
}

scaled_datasets = {}

for name, scaler in scalers.items():
    scaled_datasets[name] = {}
    for dataset_name, dataset in {
        "label": df_label,
        "onehot": df_onehot,
        "ordinal": df_ordinal,
        "binary": df_binary,
        "count": df_count,
        "target": df_target
    }.items():
        temp = dataset.copy()
        num_features = temp.select_dtypes(include=np.number).columns
        temp[num_features] = scaler.fit_transform(temp[num_features])
        scaled_datasets[name][dataset_name] = temp


print("Min-Max Scaled (Label):", scaled_datasets["minmax"]["label"].shape)
print("Max-Abs Scaled (OneHot):", scaled_datasets["maxabs"]["onehot"].shape)
print("Vector Scaled (Binary):", scaled_datasets["vector"]["binary"].shape)
print("Z-Score Scaled (Target):", scaled_datasets["zscore"]["target"].shape)
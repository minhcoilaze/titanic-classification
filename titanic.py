import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Load dữ liệu
df = pd.read_csv('train.csv')

# --- BƯỚC 1: FEATURE ENGINEERING (Thủ công trước khi vào Pipeline) ---

# Tách Title từ tên (VD: "Mr.", "Mrs.")
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# Gom nhóm các title hiếm (Don, Rev, Dr...) vào nhóm "Rare"
rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Tạo FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# --- BƯỚC 2: XỬ LÝ DỮ LIỆU THIẾU (AGE) THÔNG MINH ---
# Điền tuổi thiếu bằng trung vị (median) của từng nhóm Title
df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))

# --- BƯỚC 3: CHUẨN BỊ DỮ LIỆU CHO MODEL ---

# Chọn các features quan trọng
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize']
X = df[features]
y = df['Survived']

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BƯỚC 4: PIPELINE TỰ ĐỘNG HÓA ---

# Các cột số (Numerical)
numerical_cols = ['Age', 'Fare', 'FamilySize']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Điền median cho Fare nếu thiếu
    ('scaler', StandardScaler())
])

# Các cột phân loại (Categorical)
categorical_cols = ['Pclass', 'Sex', 'Embarked', 'Title']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Điền Embarked thiếu bằng giá trị hay gặp
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Gom lại
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Tạo Pipeline cuối cùng với Model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Train
model.fit(X_train, y_train)

# Đánh giá
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
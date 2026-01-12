import pandas as pd
import numpy as np

#đọc data
train_data = pd.read_csv('train.csv')
test_data  =pd.read_csv('test.csv')

print(f'Kich thuoc data: {train_data.shape}')
print(f'Kieu data: {train_data.info()}')


#tien xu ly data
def xu_ly_ho(data):
    #lấy họ trong tên riêng
    data['Hovip'] = data['Name'].str.extract(' ([A-Za-z]+)\.')
    Ho = ['Jonkheer', 'Countess', 'Capt', 'Sir', 'Lady', 'Don',
          'Major', 'Col', 'Rev', 'Dr', 'Dona']
    #thay thế họ dặc biệt bằng tên khác
    data['Hovip'] = data['Hovip'].replace(Ho, 'dacbiet')
    data['Hovip'] = data['Hovip'].replace('Ms', 'Miss')
    data['Hovip'] = data['Hovip'].replace('Mme', 'Mrs')
    data['Hovip'] = data['Hovip'].replace('Mlle', 'Miss')
    return data

train_data = xu_ly_ho(train_data)
test_data = xu_ly_ho(test_data)

def fsize_age(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['Age'] = data['Age'].fillna(data.groupby('Hovip')['Age'].transform('median'))
    return data

train_data = fsize_age(train_data)
test_data = fsize_age(test_data)

col_drop = ['PassengerId', 'SibSp', 'Parch', 'Cabin', 'Ticket']
cot_so = ['Fare', 'FamilySize']
cot_chu = ['Pclass', 'Sex', 'Embarked', 'Hovip']



from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

xu_ly_chu = Pipeline(steps = [
    ('buoc_1', SimpleImputer(strategy = 'most_frequent')),
    ('buoc_2', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
])
xu_ly_so = Pipeline(steps = [
    ('buoc_1', SimpleImputer(strategy = 'median')),
    ('buoc_2', StandardScaler())
])
xu_ly_ca = ColumnTransformer(transformers = [
    ('buoc_1', xu_ly_chu, cot_chu),
    ('buoc_2', xu_ly_so, cot_so)
])


from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.ensemble import RandomForestClassifier

x = train_data.drop(columns = ['Survived'])
y = train_data['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2, stratify = y)

#chọn mô hình
model = Pipeline(steps = [
    ('buoc_1', xu_ly_ca),
    ('buoc_2', RandomForestClassifier(random_state = 42))
])
chinh_sieu_ts = {
    'buoc_2__n_estimators' : [50, 70, 100, 150, 200],
    'buoc_2__max_depth' : [None, 3, 5, 7, 10, 15, 20],
    'buoc_2__min_samples_split' : [2, 5],
    'buoc_2__criterion' : ['gini']
}
luoi_tham_so = GridSearchCV(
    estimator = model,
    param_grid = chinh_sieu_ts,
    cv = 5,
    scoring = 'accuracy',
    n_jobs = -1
)
luoi_tham_so.fit(x_train, y_train)

#đánh giá mô hình
print(f"Tham số tôt nhất: {luoi_tham_so.best_params_}")
print(f"Điểm cv tôt nhất: {luoi_tham_so.best_score_:.2f}")
best_model = luoi_tham_so.best_estimator_
du_doan = best_model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report 
print(f"Độ chính xác: {accuracy_score(y_test, du_doan):.2f}")
print(f"Ma trận nhầm lẫn: {confusion_matrix(y_test, du_doan)}")
print(f"Báo cáo chi tiết: {classification_report(y_test, du_doan)}")
dd_cc = best_model.predict(test_data)


# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
from scipy import stats
import numpy as np
```


```
df = pd.read_csv("bmi.csv")
df
```

<img width="902" height="541" alt="image" src="https://github.com/user-attachments/assets/806259dc-c516-489a-87e5-0bdf4fb00249" />

```
df.head()
```

<img width="912" height="284" alt="image" src="https://github.com/user-attachments/assets/690aa3b1-d16e-454e-b30e-85b3873f6fad" />

```
df.dropna()
```

<img width="914" height="523" alt="image" src="https://github.com/user-attachments/assets/6d7ff2f3-547a-4437-9fc1-d65de8e23970" />

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Hs','Ws']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="905" height="512" alt="image" src="https://github.com/user-attachments/assets/7ef489e4-72fc-49b5-88b7-893c9ceee2e3" />

```
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df[['Hm','Wm']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="905" height="507" alt="image" src="https://github.com/user-attachments/assets/98ae337f-76c9-493d-b06b-f3dd6daf4c54" />

```
import pandas as pd
df = pd.read_csv("titanic_dataset.csv")
df.columns
```
<img width="907" height="182" alt="image" src="https://github.com/user-attachments/assets/1085e057-5fda-45cb-aab0-542f16b45e9c" />

```
df
```
<img width="1402" height="517" alt="Screenshot 2026-05-26 204037" src="https://github.com/user-attachments/assets/758ce345-fdeb-4b8c-86f8-0b5318ecc5c4" />

```
df.isnull().sum()
```
<img width="1403" height="355" alt="Screenshot 2026-05-26 204044" src="https://github.com/user-attachments/assets/70ca82dc-279a-493b-8526-391444a1aad7" />

```
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
X = df[['PassengerId','Pclass','Age','SibSp','Parch','Fare']]
y = df['Survived']
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
selector = SelectKBest(score_func=f_classif, k=4)
X_new = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']
selected_features = [selected_features[i] for i in selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="1395" height="377" alt="Screenshot 2026-05-26 204054" src="https://github.com/user-attachments/assets/b804b1c3-53b3-4ef9-a980-a74807b1bb3d" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="1401" height="395" alt="Screenshot 2026-05-26 204104" src="https://github.com/user-attachments/assets/5e0b8fd7-5d32-4b8a-bbdb-2db8fbb7d40a" />

```
tips.time.unique()
```
<img width="1398" height="122" alt="Screenshot 2026-05-26 204110" src="https://github.com/user-attachments/assets/73a07014-79a9-4ab7-a927-3bff00833a1b" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="1400" height="184" alt="Screenshot 2026-05-26 204115" src="https://github.com/user-attachments/assets/6b9b666c-6631-49a0-a0cc-f308071cbcfb" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="1402" height="168" alt="Screenshot 2026-05-26 204120" src="https://github.com/user-attachments/assets/e04ee168-90f9-49f5-be0b-68df53ada910" />


# RESULT:

Thus, the given dataset was read, cleaned, feature scaling and feature selection techniques were applied, and the processed data was saved successfully.

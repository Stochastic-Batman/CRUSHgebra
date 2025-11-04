import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


student_performance = fetch_ucirepo(name='Student Performance')
G3 = student_performance.data.targets["G3"]
romantic = student_performance.data.features["romantic"]
X = student_performance.data.features.drop(columns=["romantic"])
y = pd.DataFrame({'G3': G3, 'romantic': romantic})
og_columns = [col for col in X.columns]

# random_state for reproducibility (95 for Lightning McQueen); stratify to ensure balanced distribution for romantic
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=95, stratify=y['romantic'])

G3_train, G3_test = y_train['G3'], y_test['G3']
romantic_train, romantic_test = y_train['romantic'], y_test['romantic']

# after checking manually, it turns out that all the categorical variables are of "object" datatype.
categorical_columns = []
numerical_columns = []
for col in X_train.columns:
    if X_train[col].dtype == "object":
        categorical_columns.append(col)
    else:
        numerical_columns.append(col)

# for both categorical and numerical variables, we fit the encoder, normalizer, or standardizer on `X_train`, and then apply the fitted method to `X_test` to prevent data leakage
# ordinal categorical variables are already numerical variables
# binary categorical variables (encode as 0/1)
binary_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet']
# nominal categorical variables (one-hot encode)
nominal_columns = ['Mjob', 'Fjob', 'reason', 'guardian']

# create the preprocessor
column_transformer = ColumnTransformer(
    transformers=[
        ('binary', OrdinalEncoder(), binary_columns),
        ('nominal', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_columns),
        ('numerical', StandardScaler(), numerical_columns),
    ]
)

# fit on training data ONLY
column_transformer.fit(X_train)

# Transform both training and test sets
X_train_encoded = column_transformer.transform(X_train)
X_test_encoded = column_transformer.transform(X_test)

# converting back for easier inspection. I do not think this is necessary, but probably this will help
feature_names = (
    binary_columns +
    column_transformer.named_transformers_['nominal'].get_feature_names_out(nominal_columns).tolist() +
    numerical_columns
)

X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)


# not in Jupyter notebook:
os.makedirs('tmp', exist_ok=True)

# Save processed data for training
X_train_encoded.to_csv('tmp/X_train_encoded.csv', index=False)
X_test_encoded.to_csv('tmp/X_test_encoded.csv', index=False)
y_train.to_csv('tmp/y_train.csv', index=False)
y_test.to_csv('tmp/y_test.csv', index=False)
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

def preprocess_data(data, target_column, save_path, header_path):
    data = data.dropna(subset=[target_column])
    # Drop Fitur Student ID
    if 'student_id' in data.columns:
        data = data.drop(columns=['student_id'])
    
    # Identifikasi Fitur numeric dan kategorikal (Kecuali target 'grades')
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Pastikan target_column tidak ada di numeric_features atau categorical_features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Mendapatkan nama kolom tanpa kolom target
    column_names = data.columns.drop(target_column)

    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)

    # Menyimpan nama kolom sebagai header tanpa data
    df_header.to_csv(header_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {header_path}")

    # Pipeline untuk fitur numerik (Imputasi Median + Standard Scaling)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk fitur kategorikal (Imputasi Modus + One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Column Transformer untuk menggabungkan kedua pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # # Memisahkan target dan fitur
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Membagi data menjadi training dan testing set (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Menghapus Outlier berdasarkan IQR
    for feature in numeric_features:
        Q1 = X_train[feature].quantile(0.25)
        Q3 = X_train[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        train_mask = X_train[feature].between(lower, upper)
        test_mask = X_test[feature].between(lower, upper)

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

    # Fitting dan transformasi data pada training set
    X_train_transformed = preprocessor.fit_transform(X_train)
    # Transformasi data pada testing set
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Simpan Pipeline (.joblib) untuk digunakan di tahap selanjutnya
    dump(preprocessor, save_path)
    print(f"Pipeline preprocessing berhasil di-dump ke: {save_path}")

    return X_train_transformed, X_test_transformed, y_train, y_test, feature_names
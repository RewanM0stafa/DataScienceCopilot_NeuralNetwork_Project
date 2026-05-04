import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    تنظيف الداتا في 3 خطوات:
    1. معالجة القيم المفقودة
    2. حذف الـ Outliers (بس لو n > 50 صف)
    3. تحويل النصوص لأرقام
    """
    df_clean = df.copy()

    # ============================================================
    # الخطوة 1: معالجة القيم المفقودة
    # mean للأرقام - mode للنصوص
    # ============================================================
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() == 0:
            continue  # العمود نظيف، مفيش داعي نعمل حاجة

        if df_clean[col].dtype in ['int64', 'float64']:
            imputer = SimpleImputer(strategy='mean')
        else:
            imputer = SimpleImputer(strategy='most_frequent')

        df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()

    # ============================================================
    # الخطوة 2: حذف الـ Outliers بالـ IQR
    # ⚠️ بس لو الداتا أكبر من 50 صف
    # (علشان متكسرش الداتاسيت الصغيرة زي اللي عندنا)
    # ============================================================
    if len(df_clean) > 50:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # Clip instead of drop -> keeps all rows, just caps extreme values
            # Dropping rows risks removing entire classes from small datasets
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

        print(f"Outlier removal done: {len(df_clean)} rows remaining")
    else:
        print(f"Outlier removal skipped: only {len(df_clean)} rows (threshold is 50)")

    # ============================================================
    # الخطوة 3: Label Encoding للنصوص
    # ضروري لأن الـ Neural Network مش بتفهم نصوص
    # ============================================================
    for col in df_clean.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    print(f"Data cleaning done: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
    return df_clean


# ============================================================
# تيست سريع على الداتاست الصغيرة (8 صفوف)
# ============================================================
if __name__ == "__main__":
    sample_data = {
        'A': [10, 12, 2, 1, 11, 3, 15, 0],
        'B': [20, 22, 5, 4, 21, 6, 25, 2],
        'C': [0.5, 0.7, 0.1, 0.2, 0.6, 0.15, 0.9, 0.05],
        'target': [1, 1, 0, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(sample_data)
    print("قبل التنظيف:", df.shape)
    df_clean = clean_data(df)
    print("بعد التنظيف:", df_clean.shape)
    print(df_clean)
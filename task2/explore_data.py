import pandas as pd

# Load your dataset
file_name = 'n_meisrishvili25_87421.csv'
print(f"Loading dataset: {file_name}")
print("=" * 50)

try:
    df = pd.read_csv(file_name)

    # Basic info
    print(f"✓ Dataset loaded successfully!")
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nDataset info:")
    print(df.info())

    print(f"\nStatistical summary:")
    print(df.describe())

    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())

    # Check the target column (probably the last one or named 'label', 'spam', 'class')
    print(f"\nChecking for target column:")

    # Common target column names
    target_candidates = ['label', 'spam', 'is_spam', 'class', 'target', 'type', 'category']

    found_target = False
    for col in target_candidates:
        if col in df.columns:
            print(f"  Found: '{col}' - contains: {df[col].unique()}")
            print(f"  Value counts:")
            print(df[col].value_counts())
            found_target = True
            break

    if not found_target:
        print("  No common target column found. Checking last column...")
        last_col = df.columns[-1]
        print(f"  Last column: '{last_col}' - contains: {df[last_col].unique()[:10]}")
        print(f"  Value counts:")
        print(df[last_col].value_counts())

    # Check data types
    print(f"\nData types:")
    print(df.dtypes)

except FileNotFoundError:
    print(f"✗ Error: File '{file_name}' not found!")
    print("Make sure the file is in the current directory.")
except Exception as e:
    print(f"✗ Error: {e}")
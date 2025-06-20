import pandas as pd

def load_companies(filepath: str) -> pd.DataFrame:
    """
    Loads company data from a CSV file.

    Args:
        filepath: The path to the CSV file.

    Returns:
        A pandas DataFrame with the company data.
    """
    df = pd.read_csv(filepath)
    df['company_name'] = df['company_name'].fillna(df['domain_name'])
    df['website'] = df['domain_name'].apply(lambda x: f"http://{x}")
    return df

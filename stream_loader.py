import pandas as pd
import numpy as np

def load_stream(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Mark top 5% values as anomalies
    threshold = np.percentile(df['value'], 95)
    df['label'] = (df['value'] > threshold).astype(int)
    
    for _, row in df.iterrows():
        yield row
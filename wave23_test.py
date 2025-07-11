import pandas as pd
import numpy as np
import os
from datetime import datetime

print("Wave 23 - Testing new indicators")

# Load cached data
cache_file = "SPY_1990-01-01_2025-01-10.csv"
if os.path.exists(cache_file):
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    print(f"Data loaded: {len(df)} rows")
    
    # Test new indicator: Price acceleration
    close = df['Close']
    velocity = close.pct_change(20)
    acceleration = velocity.diff(10)
    
    print(f"Acceleration mean: {acceleration.mean():.6f}")
    print(f"Acceleration std: {acceleration.std():.6f}")
else:
    print("Cache file not found")

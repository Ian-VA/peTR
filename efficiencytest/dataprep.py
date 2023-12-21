import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("electricdata.csv")
df = df.set_index(['DATE'])
df.index = pd.to_datetime(df.index)

if not df.index.is_monotonic:
    df = df.sort_index()

df = df.rename(columns={'Date': 'Value'})

def plot(title):
    plt.plot(df.index, df.values)
    plt.title(title)
    plt.show()

plot("Electricity Prices Over Time")

    


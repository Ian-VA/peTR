import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("electricdata.csv")

def plot(title):
    plt.plot(df["DATE"], df["IPG2211A2N"])
    plt.show()

plot("Electricity Prices Over Time")

    


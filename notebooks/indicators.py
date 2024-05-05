#importing liberaries
import pandas as pd
import talib
import pynance as pn
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import numpy as np
# Load and Prepare the Data
stock_data= pd.read_csv('../data/all_stock_data1.csv')  # Assuming you have a CSV file named 'stock_data.csv'
stock_data.head()
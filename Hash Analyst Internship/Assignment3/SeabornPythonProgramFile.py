# Python script for visualizing average life expectance of continents from 1952-2007

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""Uncomment this line of code if working with Jupyter"""
# %matplotlib inline #This makes plt plots to be inline if using Jupyter notebook

df = pd.read_csv('gapminder-FiveYearData.txt')
pivot_table = pd.pivot_table(
    df, values='lifeExp', index='continent', columns='year')


plt.figure(figsize=(9, 6))
sns.heatmap(pivot_table, annot=True, linewidth=0.5, cmap='RdYlBu')
plt.show()

"""Uncomment this line of code if you wish to save the viz automatically"""
# plt.savefig('ImageFileCreatedBySeaborn') # This line of code saves the visualization as an image in PNG format

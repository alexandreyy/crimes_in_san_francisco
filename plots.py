'''
Created on 30/09/2015

@author: Alexandre Yukio Yamashita
'''

import os
import io
import codecs
import pandas as pd
import numpy as np
import string
import operator
from zipfile import ZipFile, is_zipfile

import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import contextmanager
from string import capwords

# Plotting Options
sns.set_style("whitegrid")
sns.despine()

def plot_bar(df, title, filename, color = None):
    p = (
        'Set2', 'Paired', 'colorblind', 'husl',
        'Set1', 'coolwarm', 'RdYlGn', 'spectral'
    )

    if color == None:
        color = sns.color_palette(np.random.choice(p), len(df))
    bar = df.plot(kind = 'barh',
                  title = title,
                  fontsize = 8,
                  figsize = (12, 8),
                  stacked = False,
                  width = 1,
                  color = color,
    )

    bar.figure.savefig(filename)
    plt.xlabel("Quantidade de crimes")
    plt.ylabel("Distrito")
    plt.tight_layout()

    plt.show()

def plot_top_crimes(df, column, title, fname, items = 0):
    df.columns = df.columns.map(operator.methodcaller('lower'))
    by_col = df.groupby(column)
    col_freq = by_col.size()
    col_freq.index = col_freq.index.map(capwords)
    col_freq.sort(ascending = True, inplace = True)

    if items == 0:
        items = len(col_freq)

    cmap = plt.get_cmap('gray')
    indices = np.linspace(0, cmap.N, items)
    my_colors = [cmap(int(i)) for i in indices]
    col_freq = col_freq[slice(-1, -items, -1)]
    col_freq = col_freq[::-1]
    plot_bar(col_freq, title, fname, color = my_colors)


def extract_csv(filepath):
    zp = ZipFile(filepath)
    csv = [f for f in zp.namelist() if os.path.splitext(f)[-1] == '.csv']
    return zp.open(csv.pop())

@contextmanager
def zip_csv_opener(filepath):
    fp = extract_csv(filepath) if is_zipfile(filepath) else open(filepath, 'rb')
    try:
        yield fp
    finally:
        fp.close()

def plot_top_crimes_2(df):
    path = 'resources/crimes.csv'
#
    data = pd.read_csv(path, quotechar = '"', skipinitialspace = True)
    data = data.as_matrix()

    labels = ["year", "month", "day", "time"]
    years = []
    months = []
    days = []
    times = []

    for data in data[:, 0]:
        splitted_timestamp = data.split("-")
        years.append(int(splitted_timestamp[0]))
        months.append(int(splitted_timestamp[1]))
        day_time = splitted_timestamp[2].split()
        days.append(int(day_time[0]))
        splitted_time = day_time[1].split(":")
        times.append(splitted_time[0])

    del splitted_timestamp

    times = np.array(times)
    times = times.astype(int)
    times = times.tolist()


    items = 0
    column = 'hora'
    title = 'Police Department Activity',
    fname = 'police.png'

    df.columns = df.columns.map(operator.methodcaller('lower'))
    # print times.shape

    print df.columns

    by_col = df.groupby(column)
    col_freq = by_col.size()
    col_freq.index = col_freq.index.map(capwords)

    col_freq.sort(ascending = True, inplace = True)
    plot_bar(col_freq[slice(-1, -items, -1)], title, fname)


def input_transformer(filepath):
    with zip_csv_opener(filepath) as fp:
        raw = fp.read().decode('utf-8')
    return pd.read_csv(io.StringIO(raw), parse_dates = True, index_col = 0, na_values = 'NONE')

df = input_transformer('resources/train.csv')

# plot_top_crimes(df, 'category', 'Top Crime Categories', 'category.png')
# plot_top_crimes(df, 'resolution', 'Top Crime Resolutions', 'resolution.png')
plot_top_crimes(df, 'pddistrict', 'Distrito', 'police.png')
# plot_top_crimes_2(df)

# plot_top_crimes(df, 'address', 'Enderecos com mais crimes', 'location.png', items = 10)
# plot_top_crimes(df, 'descript', 'Descriptions', 'descript.png', items = 10)

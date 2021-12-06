#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 23:22:41 2021

@author: olivialin
"""

import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import altair as alt
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


st.set_page_config(layout="wide")

st.title("Welcome to my Final Project! :car:")
st.subheader("Olivia Lin, [19621212](https://github.com/linolivia/math10finalproject)")


st.write("After a _lot_ of searching, I stumbled upon an amazing [dataset](https://www.kaggle.com/uciml/autompg-dataset) with a list of cars and their features. Here it is below:")

st.write("Looking at this dataset, I have a few questions I would like to answer. First is, is there a relationship betwen a car's power and efficiency? This 'power' will be measured two ways. My next question is, can car brands be classified through their attributes?")

df = pd.read_csv("auto-mpg.csv", na_values = "?")
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
df
df.shape


st.write("As you can see, there are some elements in this dataset that say <NA>. At first, I had assumed that the NaN values were similar to our Spotify dataset. However, with closer inspection with my errors, I noticed that our NaN values were filled with question marks (?) instead.")

df = df[df.notna().all(axis = 1)]
df


st.write(df.shape)

df4 = df.copy()

st.caption("To better clean my data, I removed any rows with at least one NaN value.")


st.header("A basic plot of our data... by you!")

st.write("To begin visualizing our data, you will be able to select your own x and y axes from our given columns, to see if there are any associations.")

yourxaxis = st.selectbox("Choose your x-axis", numeric_cols)
youryaxis = st.selectbox("Choose your y-axis", numeric_cols)

st.write("Take a look at your graph!")

scales = alt.selection_interval(bind='scales')

chart = alt.Chart(df).mark_circle().encode(
    x = yourxaxis,
    y = youryaxis,
    tooltip = "car name"
).properties(
    title = f"{yourxaxis} vs. {youryaxis}"
).add_selection(
    scales,
)
chart

st.caption("to try something [new](https://altair-viz.github.io/user_guide/interactions.html), you can scroll into a certain part of your graph for a better look.")

st.write("For me, I am particularly interested in seeing if there is an association between miles per gallon (mpg) and horsepower, as well as the relationship between acceleration and horsepower. To explore this, I created two graphs, relative to what I would like to examine. The titles on these graphs were created in reference to [this article](https://github.com/streamlit/streamlit/issues/1129).")

chart = alt.Chart(df).mark_circle().encode(
    x = "mpg:Q",
    y = "horsepower:Q",
    tooltip = "car name",
).properties(
    title = "MPG vs Horsepower"
)

chart2 = alt.Chart(df).mark_circle().encode(
    x = "acceleration",
    y = "horsepower",
    tooltip = "car name"
).properties(
    title = "Acceleration vs Horsepower"
)
chart | chart2

st.write("as you can see, there seems to be a negative association between both MPG and Horsepower, as well as Acceleration and Horsepower. This is logical, as horsepower determines the strength of your vehicle (its acceleration). The more power a vehicle takes, the more inefficient it is (lower mpg).")

st.header("scikit-learn on our data")

st.write("To cover our next question, I will cluster our data using scikit-Learn, to see if there is a relationship within these clusters.")

st.write("Because many of these columns are not in the same units of measurement, it's important to standardize the data when putting it into our machine learning algorithm.")

df2 = df[numeric_cols]
scaler = StandardScaler()
scaler.fit(df2)
df3 = pd.DataFrame(scaler.transform(df2), columns = df2.columns)
df3

st.markdown("For a fun visualization of how standardscaler works, the max cells are highlighted in a vibrant violet color, and the min in a light green. This modified dataset was created in reference to [this article](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Styler-Functions). Notice that these values are very similar throughout the columns. This is because standardscaler removes the mean and scales to unit variance (variance of 1).")



def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')
    
df3.style.apply(highlight_max, props='color:white;background-color:violet', axis=0)

def highlight_min(s, props=''):
    return np.where(s == np.nanmin(s.values), props, '')
df3.style.apply(highlight_min, props='color:white;background-color:lightgreen', axis=0)
df3

st.subheader("KMeans")

st.write("Now, we are able to cluster our data using KMeans. You are free to choose as many (or as little) clusters you would like, and notice possible overfitting that arises from the number of clusters that are chosen. The cluster number will be shown on the far right of our new dataframe.")

yourcluster = st.slider("How many clusters would you like?", min_value = 1, max_value = 392, value = 1)

kmeans = KMeans(yourcluster)

kmeans.fit(df3)

df["cluster"] = kmeans.predict(df3)
df

st.write("Just looking at the cluster numbers is difficult, so we will plot an Altair graph with colors to help create a distinction between other clusters.")

yourxaxis2 = st.selectbox("Choose your x-axis again", numeric_cols )
youryaxis2 = st.selectbox("Choose your y-axis again", numeric_cols)

stchart = alt.Chart(df).mark_circle().encode(
    x = yourxaxis2,
    y = youryaxis2,
    tooltip = "car name",
    color = "cluster:N"
).add_selection(
    scales
)
stchart

st.write("Note that as the numbers of clusters increase, the more our graph is overfitted, as adding a new point will drastically change our variance.")

st.write("It appears that there is __no__ evident relationship within these clusters.")

st.subheader("KNearestNeighbors")

st.write("Although it appears that there isn't a very strong relationship within clusters, we can still use supervised learning and KNearestNeighbors to predict car brands.")

yourneighbors = st.slider("choose the number of neighbors",1,50)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = yourneighbors)
X = df2[numeric_cols]
y = df["car name"]
clf.fit(X,y)

df4["our prediction"] = clf.predict(X)
df4

st.markdown("to visualize its predictions, feel free to explore this chart:")

yourxaxis3 = st.selectbox("Select your x-axis", numeric_cols )
youryaxis3 = st.selectbox("Select your y-axis", numeric_cols)


chart3 = alt.Chart(df4).mark_circle().encode(
    x = alt.X(yourxaxis3,scale=alt.Scale(zero=False)),
    y = alt.Y(youryaxis3,scale=alt.Scale(zero=False)),
    tooltip = ["car name", "our prediction"],
    color= alt.Color("our prediction",scale = alt.Scale(scheme ='paired'))
).add_selection(
    scales
)

chart3


st.write("Much like KMeans, KNearestNeighbors is prone to overfitting. Notice that as the number of neighbors increases, the more inaccurate our model becomes. Also, by hovering over certain points, it's clear that very little car brands match between its actual brand and its predicted brand. This shows that it is __not__ possible to determine car brand by its attributes.")

clicked = st.button("Thanks for reading! Here's a fun treat :-)")

if clicked:
    st.balloons()


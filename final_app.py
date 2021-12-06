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
#from tensorflow import keras

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

st.caption("To better clean my data, I removed any rows with at least one NaN value.")


st.header("A basic plot of our data... by you!")

st.write("To begin visualizing our data, you will be able to select your own x and y axes from our given columns, to see if there are any associations.")

yourxaxis = st.selectbox("Choose your x-axis", numeric_cols )
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
)
stchart

st.write("Note that as the numbers of clusters increase, the more our graph is overfitted, as adding a new point will drastically change our variance.")

st.write("It appears that there is __no__ evident relationship within these clusters.")

st.header("Keras and Car Brands")

st.write("Here I will answer my last question, if it's possible to determine what car brand a vehicle is based on its attributes. To do so, I will need to find a way to extract the car brand name from the `car name` column.")

st.write("to create this dropdown box of car brands while avoiding repeated car brand names, I referenced [this article](https://www.geeksforgeeks.org/python-removing-duplicates-from-tuple/#:~:text=Method%20%231%20%3A%20Using%20set(),back%20again%20using%20tuple()%20.).")

df["car name"] = df["car name"].astype('str').str.split(" ")
brands = tuple(set(tuple([df.loc[:,"car name"][x][0] for x in df.index])))

yourbrand = st.selectbox("What car brand would you like to test?", brands)

df2 = df.copy()

df2[f"is_{yourbrand}"] = df2.loc[:,"car name"].map(lambda is_brand: yourbrand in is_brand)
df2
brandsum = df2[f"is_{yourbrand}"].sum(axis = 0)

st.write(f"{brandsum} {yourbrand} cars are in this dataset.")

st.write("Please refer to the [source code](https://github.com/linolivia/math10finalproject/blob/main/final_app.py) to examine my work with keras.")
#the success of the following machine learning exercise has been depending solely on its performance on Google Colab. I have been working on this project on an M1 macbook, and so downloading Tensorflow has so far been unsuccessful.

#X_train = df[numeric_cols]
#X_train = np.asarray(X_train).astype(np.float32)

#y_train = df[f"is_{yourbrand}"]
#y_train = np.asarray(y_train).astype(np.float32)

#model = keras.Sequential(
    #[
#        keras.layers.InputLayer(input_shape = (9,)), # shape is 9 because there are nine numeric columns in my dataset.
        #keras.layers.Flatten(),
#        keras.layers.Dense(16, activation="sigmoid"),
#        keras.layers.Dense(16, activation="sigmoid"),
#        keras.layers.Dense(1,activation="sigmoid") #output is 1 because its decisions are binary; either the car is your selected car brand, or it is not.
#    ]
#)
#
#model.compile(
#    loss="binary_crossentropy",
#    optimizer=keras.optimizers.SGD(learning_rate=0.01),
#    metrics=["accuracy"],
#)
#
#history = model.fit(X_train,y_train,epochs=100) #100 iterations

st.write("As you can see, the accuracy hovers at around 85% for whichever car brand is selected. This is not necessarily a good thing; however. These algorithms are prone to overfitting because it's possible the algorithm is simply memorizing all of the data. So, take these conclusions with a grain of salt. Because of this limitation that we are aware of, I would conclude that it is __not__ possible to determine the car brand from its attributes.  ")

clicked = st.button("Thanks for reading! Here's a fun treat :-)")

if clicked:
    st.balloons()


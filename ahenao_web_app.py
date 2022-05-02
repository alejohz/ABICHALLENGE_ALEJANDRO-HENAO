import pandas as pd
import numpy as np
import streamlit as st
import json

from sagemaker.session import Session

import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import xgboost as xgb
from sm_model import sagemaker_xgboost
import sagemaker


# TODO: Read data from S3
iris_data = load_iris()
# separate the data into features and target
features = iris_data.feature_names
y = iris_data.target

model = sagemaker_xgboost()

def construct_sidebar():

    cols = [col for col in features.columns]

    st.sidebar.markdown(
        '<p class="header-style">Iris Data Classification</p>',
        unsafe_allow_html=True
    )
    sepal_length = st.sidebar.selectbox(
        f"Select {cols[0]}",
        sorted(features[0].unique())
    )

    sepal_width = st.sidebar.selectbox(
        f"Select {cols[1]}",
        sorted(features[1].unique())
    )

    petal_length = st.sidebar.selectbox(
        f"Select {cols[2]}",
        sorted(features[2].unique())
    )

    petal_width = st.sidebar.selectbox(
        f"Select {cols[3]}",
        sorted(features[3].unique())
    )
    values = [sepal_length, sepal_width, petal_length, petal_width]

    return values

def plot_pie_chart(probabilities):
    fig = go.Figure(
        data=[go.Pie(
                labels=list(iris_data.target_names),
                values=probabilities
        )]
    )
    fig = fig.update_traces(
        hoverinfo='label+percent',
        textinfo='value',
        textfont_size=15
    )
    return fig




def construct_app():
    st.header("AB in BEV web app deployment of sagemaker model")


    if st.checkbox('DEPLOY'):
        with st.spinner("Training ongoing"):
            if not hasattr(st, 'predictor'):
                st.predictor = model.model_deploy()
            predictor = st.predictor



    if st.button('CANCEL'):
        model.model_cancel()
        st.stop()
    values = construct_sidebar()
    if st.checkbox('Calculate'):


        r = model.model_predict(values, st.predictor)
        r = r.decode()
        r = r.split(",")
        r = [float(a) for a in r]
        st.text('This is the prediction from XGBoost model')
        st.text(r)
        st.text('This are the values inputted for the prediction')
        st.text(values)
        p1 = r[0]
        p2 = r[1]
        p3 = r[2]
        probabilities = [p1, p2, p3]
        classes = ['Setosa', 'Versicolour', 'Virginica']
        prediction_str = classes[np.argmax(probabilities)]
        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-style"> Iris Data Predictions </p>',
            unsafe_allow_html=True
        )

        column_1, column_2 = st.columns(2)
        column_1.markdown(
            f'<p class="font-style" >Prediction </p>',
            unsafe_allow_html=True
        )
        column_1.write(f"{prediction_str}")

        column_2.markdown(
            '<p class="font-style" >Probability </p>',
            unsafe_allow_html=True
        )
        column_2.write(f"{probabilities[0]}")

        fig = plot_pie_chart(probabilities)
        st.markdown(
            '<p class="font-style" >Probability Distribution</p>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    construct_app()


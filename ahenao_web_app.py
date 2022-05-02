# Importing useful libraries

import pandas as pd
import numpy as np
import streamlit as st
import json

import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sm_model import sagemaker_xgboost


# TODO: 
# 
# Read data from S3
# Define a class for the wep app
iris_data = load_iris()
# separate the data into features and target
features = pd.DataFrame(
    iris_data.data, columns=iris_data.feature_names
)
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
        sorted(features[cols[0]].unique())
    )

    sepal_width = st.sidebar.selectbox(
        f"Select {cols[1]}",
        sorted(features[cols[1]].unique())
    )

    petal_length = st.sidebar.selectbox(
        f"Select {cols[2]}",
        sorted(features[cols[2]].unique())
    )

    petal_width = st.sidebar.selectbox(
        f"Select {cols[3]}",
        sorted(features[cols[3]].unique())
    )
    values = [sepal_length, sepal_width, petal_length, petal_width]

    return values

def plot_pie_chart(probabilities):
    """
    Function to plot and show probability distribution of the 3 classes:
    Params:
    Probabilites list
        3 floating numbers indicating probability of the sample belonging to the index class
    """
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
    st.session_state.load_state=False

    st.text('Check Deploy checkbox for model deploying')
    if st.checkbox('DEPLOY'):
        if not st.session_state.load_state:
            with st.spinner("Training ongoing"):
                if not hasattr(st, 'predictor'):
                    st.predictor = model.model_deploy()
                    st.text('MODEL DEPLOYED')
                    st.session_state.load_state=True
    st.text('Remember to cancel the endpoint')
    if st.button('CANCEL'):
        model.model_cancel(st.predictor)
        st.session_state.load_state = False
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


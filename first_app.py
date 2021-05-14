import json
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

days = ["Monday", "Tuesday", "Wednesday", "Thursay", "Friday", "Saturday", "Sunday"]

st.set_page_config(
    page_title="BikesVsTaxisNYC",
    layout="wide",
)

st.title("Bike vs Taxis")

st.write("## Intro")

with open("datasets/NYC_Taxi_Zones.geojson") as json_file:
    zone_data = json.load(json_file)
for d in zone_data["features"]:
    d["id"] = d["properties"]["location_id"]


def load_data(path):
    return pd.read_csv(path)


@st.cache()
def load_datasets():
    df_taxi = load_data("datasets/taxi_ENDzone_speed_by_hour.csv").drop(
        columns=["Unnamed: 0"]
    )
    df_bikes = load_data("datasets/bike_zone_speed_by_hour.csv").drop(
        columns=["Unnamed: 0"]
    )
    return df_taxi, df_bikes


@st.cache(hash_funcs={dict: lambda _: None})
def create_choropleths(hour):
    fig_taxi = px.choropleth_mapbox(
        df_taxi[df_taxi.Hour == hour],
        geojson=zone_data,
        locations="end_station",
        color="avg_speed",
        color_continuous_scale="Viridis",
        range_color=(0, df_taxi.avg_speed.max()),
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": lat, "lon": lon},
        opacity=0.5,
        labels={"Number": "random number"},
    )
    fig_taxi.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig_bikes = px.choropleth_mapbox(
        df_bikes[df_bikes.Hour == hour],
        geojson=zone_data,
        locations="start_station",
        color="avg_speed",
        color_continuous_scale="Viridis",
        range_color=(0, df_bikes.avg_speed.max()),
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": lat, "lon": lon},
        opacity=0.5,
        labels={"Number": "random number"},
    )
    fig_bikes.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig_taxi, fig_bikes


# Run counts for subscriber
@st.cache()
def load_counts():
    with open("datasets/costumer_dict.pickle", "rb") as handle:
        costumer_dict = pickle.load(handle)
    with open("datasets/subscriber_dict.pickle", "rb") as handle:
        subscriber_dict = pickle.load(handle)
    with open("datasets/taxi_dict.pickle", "rb") as handle:
        taxi_dict = pickle.load(handle)
    return costumer_dict, subscriber_dict, taxi_dict


@st.cache(hash_funcs={dict: lambda _: None})
def get_hourly_counts_figures():
    data_taxi = pd.DataFrame(
        zip(list(range(24)), taxi_counts["hour_count"]),
        columns=["hours", "count"],
        index=None,
    )
    data_customer = pd.DataFrame(
        zip(list(range(24)), customer_counts["hour_count"]),
        columns=["hours", "count"],
        index=None,
    )
    data_subscriber = pd.DataFrame(
        zip(list(range(24)), subscriber_counts["hour_count"]),
        columns=["hours", "count"],
        index=None,
    )
    fig_taxi = px.bar(data_taxi, x="hours", y="count")
    fig_taxi.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})
    fig_customers = px.bar(
        data_customer,
        x="hours",
        y="count",
    )
    fig_customers.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})
    fig_subscribers = px.bar(
        data_subscriber,
        x="hours",
        y="count",
    )
    fig_subscribers.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})

    return fig_taxi, fig_customers, fig_subscribers


@st.cache(hash_funcs={dict: lambda _: None})
def get_DOW_counts_figures():
    data_taxi = pd.DataFrame(
        zip(days, taxi_counts["DOW_count"]),
        columns=["Day of the Week", "count"],
        index=None,
    )
    data_customer = pd.DataFrame(
        zip(days, customer_counts["DOW_count"]),
        columns=["Day of the Week", "count"],
        index=None,
    )
    data_subscriber = pd.DataFrame(
        zip(days, subscriber_counts["DOW_count"]),
        columns=["Day of the Week", "count"],
        index=None,
    )
    fig_taxi = px.bar(data_taxi, x="Day of the Week", y="count")
    fig_taxi.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})
    fig_customers = px.bar(
        data_customer,
        x="Day of the Week",
        y="count",
    )
    fig_customers.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})
    fig_subscribers = px.bar(
        data_subscriber,
        x="Day of the Week",
        y="count",
    )
    fig_subscribers.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})

    return fig_taxi, fig_customers, fig_subscribers


@st.cache(hash_funcs={dict: lambda _: None})
def get_monthly_counts_figures():
    data_taxi = pd.DataFrame(
        zip(months, taxi_counts["month_count"]),
        columns=["months", "count"],
        index=None,
    )
    data_customer = pd.DataFrame(
        zip(months, customer_counts["month_count"]),
        columns=["months", "count"],
        index=None,
    )
    data_subscriber = pd.DataFrame(
        zip(months, subscriber_counts["month_count"]),
        columns=["months", "count"],
        index=None,
    )
    fig_taxi = px.bar(data_taxi, x="months", y="count")
    fig_taxi.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})
    fig_customers = px.bar(
        data_customer,
        x="months",
        y="count",
    )
    fig_customers.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})
    fig_subscribers = px.bar(
        data_subscriber,
        x="months",
        y="count",
    )
    fig_subscribers.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})

    return fig_taxi, fig_customers, fig_subscribers


@st.cache(hash_funcs={dict: lambda _: None})
def get_HOW_counts_figures():
    data_taxi = pd.DataFrame(
        zip(list(range(168)), taxi_counts["HOW_count"]),
        columns=["Hour of Week", "count"],
        index=None,
    )
    data_customer = pd.DataFrame(
        zip(list(range(168)), customer_counts["HOW_count"]),
        columns=["Hour of Week", "count"],
        index=None,
    )
    data_subscriber = pd.DataFrame(
        zip(list(range(168)), subscriber_counts["HOW_count"]),
        columns=["Hour of Week", "count"],
        index=None,
    )
    fig_taxi = px.bar(data_taxi, x="Hour of Week", y="count")
    fig_taxi.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})
    fig_customers = px.bar(
        data_customer,
        x="Hour of Week",
        y="count",
    )
    fig_customers.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})
    fig_subscribers = px.bar(
        data_subscriber,
        x="Hour of Week",
        y="count",
    )
    fig_subscribers.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})

    return fig_taxi, fig_customers, fig_subscribers


customer_counts, subscriber_counts, taxi_counts = load_counts()

df_taxi, df_bikes = load_datasets()

max_speed = df_taxi.avg_speed.max()

hour = st.slider("Hour of the Day", 0, 23, 17)  # 👈 this is a widget
lat = 40.760610
lon = -73.95242


fig_taxi, fig_bikes = create_choropleths(hour)


st.write("## Average Travel Speed")
st.write("Random text")
left_column, right_column = st.beta_columns(2)
left_column.write("## Taxi")
left_column.plotly_chart(fig_taxi)
right_column.write("## Bikes")
right_column.plotly_chart(fig_bikes)

st.write("## Descriptive statistics")
add_selectbox = st.selectbox(
    "Choose Timeframe", ("Hours", "Month", "Day of the Week", "Hour of the Week")
)
left_column_1, middle_column_1, right_column_1 = st.beta_columns((1, 1, 1))
# You can use a column just like st.sidebar1
(
    taxi_hourly_counts,
    customer_hourly_counts,
    subscriber_hourly_counts,
) = get_hourly_counts_figures()
taxi_month, customer_month, subscriber_month = get_monthly_counts_figures()
taxi_HOW, customer_HOW, subscriber_HOW = get_HOW_counts_figures()
taxi_DOW, customer_DOW, subscriber_DOW = get_DOW_counts_figures()

left_column_1.write("Taxi Rides")
middle_column_1.write("Subscriber Rides")
right_column_1.write("Customer Rides")

if add_selectbox == "Month":
    left_column_1.plotly_chart(taxi_month)
    middle_column_1.plotly_chart(customer_month)
    right_column_1.plotly_chart(subscriber_month)
elif add_selectbox == "Hours":
    left_column_1.plotly_chart(taxi_hourly_counts)
    middle_column_1.plotly_chart(customer_hourly_counts)
    right_column_1.plotly_chart(subscriber_hourly_counts)
elif add_selectbox == "Day of the Week":
    left_column_1.plotly_chart(taxi_DOW)
    middle_column_1.plotly_chart(customer_DOW)
    right_column_1.plotly_chart(subscriber_DOW)
else:
    left_column_1.plotly_chart(taxi_HOW)
    middle_column_1.plotly_chart(customer_HOW)
    right_column_1.plotly_chart(subscriber_HOW)


st.write("## Predictive Modeling")
# Forms can be declared using the 'with' syntax
left_column_3, right_column = st.beta_columns((2, 3))
with left_column_3:
    with st.form(key="my_form"):
        name = st.text_input(label="Enter your name")
        date = st.selectbox("Enter Month", months)
        hour = st.selectbox("Enter hour", list(range(0, 24)))
        day_of_the_week = st.selectbox("Enter day of the week", days)
        starting_location = st.text_input(
            label="Enter your current location (long/lat)"
        )
        target_location = st.text_input(label="Enter where you are (long/lat)")
        submit_button = st.form_submit_button(label="Submit")
    # st.form_submit_button returns True upon form submit
st.write("### Model Prediction")
if submit_button:
    st.write(f"  Hello {name}, have a nice bike trip!")

right_column.write(
    """
        ## What is our model?
        What is it doing?\n
        What it returns? 
        """
)

st.write("## Route Network Analysis for Bikes")

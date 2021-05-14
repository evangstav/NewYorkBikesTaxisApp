import json
import pickle

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="BikesVsTaxisNYC",
    layout="wide",
)

PLACE_HOLDER_DISTANCE = 3000

MONTHS = [
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

month_to_index = dict([(m, i) for i, m in enumerate(MONTHS)])

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursay", "Friday", "Saturday", "Sunday"]
days_to_index = dict([(d, i) for i, d in enumerate(DAYS)])

# NYC location
LAT = 40.760610
LON = -73.95242


def get_avg_speed(df, starting_zone, ending_zone, hour):
    avg_speed = df["avg_speed"][
        (df["start_station"] == starting_zone)
        & (df["end_station"] == ending_zone)
        & (df["Hour"] == hour)
    ]
    if len(avg_speed) == 0:
        avg_speed = df["avg_speed"][
            (df["end_station"] == ending_zone) & (df["Hour"] == hour)
        ]
    if len(avg_speed) == 0:
        avg_speed = df["avg_speed"][(df["Hour"] == hour)]
    if len(avg_speed) == 0:
        avg_speed = df_taxi["avg_speed"]
    avg_speed = np.mean(avg_speed)

    return avg_speed


def load_data(path):
    return pd.read_csv(path)


@st.cache()
def load_datasets():
    df_taxi = load_data("datasets/taxi_zone_speed_by_hour.csv").drop(
        columns=["Unnamed: 0"]
    )
    df_bikes = load_data("datasets/bike_zone_speed_by_hour.csv").drop(
        columns=["Unnamed: 0"]
    )
    return df_taxi, df_bikes


@st.cache
def load_models():
    """
    returns: (bike model, taxi model, taxi price model)
    """
    # LOADING the models
    rf_bikes = joblib.load("models/rf_bike_regressor2.pkl")
    rf_taxi = joblib.load("models/rf_taxi_regressor_2.pkl")
    rf_taxi_price = joblib.load("models/rf_taxi_price_regressor.pkl")
    return rf_bikes, rf_taxi, rf_taxi_price


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
        center={"lat": LAT, "lon": LON},
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
        center={"lat": LAT, "lon": LON},
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
        zip(DAYS, taxi_counts["DOW_count"]),
        columns=["Day of the Week", "count"],
        index=None,
    )
    data_customer = pd.DataFrame(
        zip(DAYS, customer_counts["DOW_count"]),
        columns=["Day of the Week", "count"],
        index=None,
    )
    data_subscriber = pd.DataFrame(
        zip(DAYS, subscriber_counts["DOW_count"]),
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
        zip(MONTHS, taxi_counts["month_count"]),
        columns=["months", "count"],
        index=None,
    )
    data_customer = pd.DataFrame(
        zip(MONTHS, customer_counts["month_count"]),
        columns=["months", "count"],
        index=None,
    )
    data_subscriber = pd.DataFrame(
        zip(MONTHS, subscriber_counts["month_count"]),
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


# LOADING THE DATA
with open("datasets/NYC_Taxi_Zones.geojson") as json_file:
    zone_data = json.load(json_file)
for d in zone_data["features"]:
    d["id"] = d["properties"]["location_id"]

customer_counts, subscriber_counts, taxi_counts = load_counts()

df_taxi, df_bikes = load_datasets()

with open("datasets/weather_dict.pkl", "rb") as f:
    weather_dict = pickle.load(f)

## LOADING MODELS


max_speed = df_taxi.avg_speed.max()
rf_bike, rf_taxi, rf_taxi_price = load_models()

st.title("Bike vs Taxis")

st.write("## Intro")


hour = st.slider("Hour of the Day", 0, 23, 17)  # 👈 this is a widget
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
# Our models take as input ['WND','CIG','VIS','TMP','DEW','SLP','distance_meters','Hour','Weekday','Month','avg_speed']
start_stations = df_taxi.start_station.unique()
end_stations = df_taxi.end_station.unique()
# Forms can be declared using the 'with' syntax
left_column_3, right_column = st.beta_columns((2, 3))
with left_column_3:
    with st.form(key="my_form"):
        name = st.text_input(label="Enter your name")
        method = st.selectbox("Are you using Bike or Taxi", ["taxi", "bike"])
        month = st.selectbox("Enter Month", MONTHS)
        hour = st.selectbox("Enter hour", list(range(0, 24)))
        weekday = st.selectbox("Enter day of the week", DAYS)
        starting_zone = st.selectbox("Enter your starting zone", start_stations)
        ending_zone = st.selectbox("Enter your end zone", end_stations)
        submit_button = st.form_submit_button(label="Submit")

    # st.form_submit_button returns True upon form submit
st.write("### Model Prediction")

distance = PLACE_HOLDER_DISTANCE

if submit_button:
    month = month_to_index[month]
    taxi_features = np.array(
        [
            weather_dict[month]["WND"],
            weather_dict[month]["CIG"],
            weather_dict[month]["VIS"],
            weather_dict[month]["TMP"],
            weather_dict[month]["DEW"],
            weather_dict[month]["SLP"],
            distance,
            hour,
            days_to_index[weekday],
            month,
            get_avg_speed(df_taxi, starting_zone, ending_zone, hour),
        ]
    ).reshape(1, -1)
    bike_features = np.array(
        [
            weather_dict[month]["WND"],
            weather_dict[month]["CIG"],
            weather_dict[month]["VIS"],
            weather_dict[month]["TMP"],
            weather_dict[month]["DEW"],
            weather_dict[month]["SLP"],
            distance,
            hour,
            days_to_index[weekday],
            month,
            get_avg_speed(df_bikes, starting_zone, ending_zone, hour),
        ]
    ).reshape(1, -1)

    time_taxi_prediction = rf_taxi.predict(taxi_features)
    price_taxi_prediction = rf_taxi_price.predict(
        np.array([distance, time_taxi_prediction[0]]).reshape(1, -1)
    )
    time_bike_prediction = rf_bike.predict(bike_features)

    st.write(
        f"  Hello {name}! \n Taking a taxi you would need {time_taxi_prediction[0]/60:.1f} minutes to reach your destination, and it will cost you around {price_taxi_prediction[0]:.2f}\$. \nTaking the bike costs 3$/h and it will take your {time_bike_prediction[0]/60:.1f} minutes!"
    )

# print(
#     f"  Hello {name}! \nTaking a taxi you would need {time_taxi_prediction[0]/60:.1f} minutes to reach your destination, and it will cost you around {price_taxi_prediction[0]:.2f}$. \nTaking the bike costs 3$/h and it will take your {time_bike_prediction[0]/60:.1f} minutes!"
# )


right_column.write(
    """
        ## What is our model?
        What is it doing?\n
        What it returns? 
        """
)
st.write("## Route Network Analysis for Bikes")

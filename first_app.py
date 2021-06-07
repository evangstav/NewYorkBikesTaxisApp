import json
import pickle

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(
    page_title="BikesVsTaxisNYC",
    layout="wide",
)

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

output_text = ""
# NYC location
LAT = 40.760610
LON = -73.95242


@st.cache
def load_images():
    return Image.open("datasets/pic_1.png"), Image.open("datasets/pic_2.png")


def get_avg_distance(df, starting_zone, ending_zone, hour):
    avg_distance = df["distance"][
        (df["start_station"] == starting_zone)
        & (df["end_station"] == ending_zone)
        & (df["Hour"] == hour)
    ]
    if len(avg_distance) == 0:
        avg_distance = df["distance"][
            (df["end_station"] == ending_zone) & (df["Hour"] == hour)
        ]
    if len(avg_distance) == 0:
        avg_distance = df["distance"][(df["Hour"] == hour)]
    if len(avg_distance) == 0:
        avg_distance = df_taxi["distance"]
    avg_distance = np.mean(avg_distance)
    return avg_distance


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


@st.cache(suppress_st_warning=True)
def load_datasets():
    df_taxi = load_data("datasets/taxi_zone_speed_DISTANCE_by_hour.csv").drop(
        columns=["Unnamed: 0"]
    )
    df_bikes = load_data("datasets/bike_zone_speed_DISTANCE_by_hour.csv").drop(
        columns=["Unnamed: 0"]
    )
    return df_taxi, df_bikes


@st.cache(suppress_st_warning=True)
def load_datasets_choro():
    df_taxi = load_data("datasets/final_taxi.csv").drop(columns=["Unnamed: 0"])
    df_bikes = load_data("datasets/final_bike.csv").drop(columns=["Unnamed: 0"])
    return df_taxi, df_bikes


@st.cache(suppress_st_warning=True)
def load_models():
    """
    returns: (bike model, taxi model, taxi price model)
    """
    # LOADING the models
    rf_bikes = joblib.load("models/rf_bike_regressor3.pkl")
    rf_taxi = joblib.load("models/rf_taxi_regressor_3.pkl")
    rf_taxi_price = joblib.load("models/rf_taxi_price_regressor.pkl")
    return rf_bikes, rf_taxi, rf_taxi_price


@st.cache(hash_funcs={dict: lambda _: None})
def create_choropleths(hour):
    fig_taxi = px.choropleth_mapbox(
        df_taxi_choro[df_taxi_choro.Hour == hour],
        geojson=zone_data,
        locations="end_station",
        color="relative_avg_zone_speed",
        color_continuous_scale="rainbow",
        range_color=(0, df_taxi_choro.relative_avg_zone_speed.max()),
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": LAT, "lon": LON},
        opacity=0.5,
        labels={"Number": "random number"},
    )
    fig_taxi.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig_bikes = px.choropleth_mapbox(
        df_bikes_choro[df_bikes_choro.Hour == hour],
        geojson=zone_data,
        locations="end_station",
        color="relative_avg_zone_speed",
        color_continuous_scale="rainbow",
        range_color=(0, df_bikes_choro.relative_avg_zone_speed.max()),
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": LAT, "lon": LON},
        opacity=0.5,
        labels={"Number": "random number"},
    )
    fig_bikes.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig_taxi, fig_bikes


@st.cache(hash_funcs={dict: lambda _: None})
def avg_hourly_relative_speed():
    relative_speed_df = pd.read_csv("datasets/relative_speed_per_hour.csv")
    fig = px.line(relative_speed_df, x="Hour", y="relative_speed", color="type")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


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
def get_monthly_counts_line():
    data_taxi = pd.DataFrame(
        zip(MONTHS, taxi_counts["month_count"]),
        columns=["months", "count"],
        index=None,
    )
    data_taxi["type"] = "taxi"
    data_customer = pd.DataFrame(
        zip(MONTHS, customer_counts["month_count"]),
        columns=["months", "count"],
        index=None,
    )
    data_customer["type"] = "customer"
    data_subscriber = pd.DataFrame(
        zip(MONTHS, subscriber_counts["month_count"]),
        columns=["months", "count"],
        index=None,
    )
    data_subscriber["type"] = "subscriber"
    data = pd.concat([data_taxi, data_subscriber, data_customer])
    fig = px.line(
        data,
        x="months",
        y="count",
        color="type",
        width=1000,
        height=400,
    )
    fig.update_layout(margin={"r": 1, "t": 0, "l": 1, "b": 0})
    return fig


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
    fig_taxi.update_layout(margin={"r": 2, "t": 0, "l": 1, "b": 0})
    fig_customers = px.bar(
        data_customer,
        x="Hour of Week",
        y="count",
    )
    fig_customers.update_layout(margin={"r": 2, "t": 0, "l": 1, "b": 0})
    fig_subscribers = px.bar(
        data_subscriber,
        x="Hour of Week",
        y="count",
    )
    fig_subscribers.update_layout(margin={"r": 2, "t": 0, "l": 1, "b": 0})

    return fig_taxi, fig_customers, fig_subscribers


# LOADING THE DATA
with open("datasets/NYC_Taxi_Zones.geojson") as json_file:
    zone_data = json.load(json_file)
for d in zone_data["features"]:
    d["id"] = d["properties"]["location_id"]
customer_counts, subscriber_counts, taxi_counts = load_counts()
df_taxi, df_bikes = load_datasets()
df_taxi_choro, df_bikes_choro = load_datasets_choro()

with open("datasets/weather_dict.pkl", "rb") as f:
    weather_dict = pickle.load(f)

image_1, image_2 = load_images()

## LOADING MODELS
rf_bike, rf_taxi, rf_taxi_price = load_models()


## STARTING PAGE
st.title("Bike vs Taxis in New York City")
st.write("# Introduction")
st.markdown(
    "This website aims to shed light on the benefits of commuting by bike in New York City relative to taking a cab. To do this the CitiBike data and Green Taxi Trip data will be used in a graphical and interactive manner. The full analysis and code are contained in this [jupyter notebook](https://colab.research.google.com/drive/1kTbYnSqxV0nabAc1PIaRHHLzNUPhEoGN?usp=sharing)"
)
st.write("# Descriptive statistics")
st.write(
    "Lets begin with some basic counts over time for different time periods. You can use the dropdown list to change the timeframe."
)
add_selectbox = st.selectbox(
    "Choose Timeframe", ("Hourly", "Monthly", "Day of the Week", "Hour of the Week")
)
left_column_1, middle_column_1, right_column_1 = st.beta_columns((1, 1, 1))
# You can use a column just like st.sidebar1
(
    taxi_hourly_counts,
    customer_hourly_counts,
    subscriber_hourly_counts,
) = get_hourly_counts_figures()
taxi_month, customer_month, subscriber_month = get_monthly_counts_figures()
line_figure = get_monthly_counts_line()
taxi_HOW, customer_HOW, subscriber_HOW = get_HOW_counts_figures()
taxi_DOW, customer_DOW, subscriber_DOW = get_DOW_counts_figures()

left_column_1.write("           Taxi Rides")
middle_column_1.write("          Customer Rides")
right_column_1.write("         Subscriber Rides")

if add_selectbox == "Monthly":
    left_column_1.plotly_chart(taxi_month, width=300)
    middle_column_1.plotly_chart(customer_month, width=300)
    right_column_1.plotly_chart(subscriber_month, width=300)
elif add_selectbox == "Hourly":
    left_column_1.plotly_chart(taxi_hourly_counts, width=300)
    middle_column_1.plotly_chart(customer_hourly_counts, width=300)
    right_column_1.plotly_chart(subscriber_hourly_counts, width=300)
elif add_selectbox == "Day of the Week":
    left_column_1.plotly_chart(taxi_DOW, width=300)
    middle_column_1.plotly_chart(customer_DOW, width=300)
    right_column_1.plotly_chart(subscriber_DOW, width=300)
else:
    left_column_1.plotly_chart(taxi_HOW, width=300)
    middle_column_1.plotly_chart(customer_HOW, width=300)
    right_column_1.plotly_chart(subscriber_HOW, width=300)
st.write(
    'The different plots reveal that a lot of people actually choose to commute by bike back and fourth from their 9-5 jobs while " Customers " are most likely tourist that cruise the city in weekends and in a more bell-curved manner through out the day.'
)
left_column_2, right_column_2 = st.beta_columns((1, 1))
right_column_2.plotly_chart(line_figure)
left_column_2.write("Lets take a close look at the usage over a year.")
left_column_2.write(
    "When you look at the different types of transportation \nover the months the plot shows that biking is more preferable \nin the warmer months and actually affects the demand for taxis."
)


st.write("# Average Travel Speed")
st.write(
    "New York City is a complicated and a big city, the taxi company make use of certain zones to represent the different areas. We took an advantage of this and visualized the average speed traveld within each zone at a given hour."
)
st.write(
    "By looking at the average speed per zone for each hour it is easier to understand how traffic affects the two different transportation options. The slider below changes the plots on the 24 hour scale. Playing arround with this reveals some interesting features, can you see them? The decrease of average speed during rush hour is much greater for the taxis that for bikes. This indicates that bikeing is a more stable and reliable mode of transportation, even when your are in a hurry!"
)
hour = st.slider("Hour of the Day", 0, 23, 17)  # 👈 this is a widget
fig_taxi, fig_bikes = create_choropleths(hour)
left_column, right_column = st.beta_columns(2)
left_column.write("## Taxi")
left_column.plotly_chart(fig_taxi)
right_column.write("## Bikes")
right_column.plotly_chart(fig_bikes)
(
    left,
    right,
) = st.beta_columns((2, 6))
st.empty()
right.plotly_chart(avg_hourly_relative_speed())
left.write(
    "## This can be confirmed with this plot over average speed per hour for the whole dataset."
)
left.write(
    "However it is interesting to see that biking travel speed \nare actually correlated with the taxis indicating that\n the bikers are dependant on the infrastructure of the taxis!"
)


st.write("# Predictive Modeling")
# Our models take as input ['WND','CIG','VIS','TMP','DEW','SLP','distance_meters','Hour','Weekday','Month','avg_speed']
start_stations = df_taxi.start_station.unique()
end_stations = df_taxi.end_station.unique()
# Forms can be declared using the 'with' syntax
left_column_3, right_column_3 = st.beta_columns((2, 3))
with left_column_3:
    with st.form(key="my_form"):
        name = st.text_input(label="Enter your name")
        month = st.selectbox("Enter Month", MONTHS)
        hour = st.selectbox("Enter hour", list(range(0, 24)))
        weekday = st.selectbox("Enter day of the week", DAYS)
        starting_zone = st.selectbox("Enter your starting zone", sorted(start_stations))
        ending_zone = st.selectbox("Enter your end zone", sorted(end_stations))
        submit_button = st.form_submit_button(label="Submit")

    # st.form_submit_button returns True upon form submit

if submit_button:
    month = month_to_index[month]
    taxi_distance = get_avg_distance(df_taxi, starting_zone, ending_zone, hour)
    taxi_features = np.array(
        [
            weather_dict[month]["WND"],
            weather_dict[month]["CIG"],
            weather_dict[month]["VIS"],
            weather_dict[month]["TMP"],
            weather_dict[month]["DEW"],
            weather_dict[month]["SLP"],
            get_avg_distance(df_taxi, starting_zone, ending_zone, hour),
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
            get_avg_distance(df_bikes, starting_zone, ending_zone, hour),
            hour,
            days_to_index[weekday],
            month,
            get_avg_speed(df_bikes, starting_zone, ending_zone, hour),
        ]
    ).reshape(1, -1)

    time_taxi_prediction = rf_taxi.predict(taxi_features)
    price_taxi_prediction = rf_taxi_price.predict(
        np.array([taxi_distance, time_taxi_prediction[0]]).reshape(1, -1)
    )
    time_bike_prediction = rf_bike.predict(bike_features)
    output_text = f"## Hello {name}! \n## Taking a taxi you would need **{time_taxi_prediction[0]/60:.1f}** minutes to reach zone {ending_zone} costing you around _{price_taxi_prediction[0]:.2f}\$_. \n## Taking the bike costs _3.5$/h_ and it will take you **{time_bike_prediction[0]/60:.1f}** minutes!"

right_column_3.write(
    "## But if you get caught in a situation where you need to get from a-b, do you know how long it will take you with the taxi compared to a bike? What about the cost? "
)
right_column_3.write(
    "This can be a problem that is hard to estimate on your own. That is why we have developed a machine learning model that can estimate those factors for you. Now you can take a well informed decisons every time and you will be surprised to find out that often times taking the bike is also quicker."
)

# right_column_3.write("## Model Prediction")
right_column_3.write(output_text)

st.write("# Route Network Analysis for Bikes")

left_column_4, right_column_4 = st.beta_columns((2, 1))
with left_column_4:
    HtmlFile = open("datasets/route_plot.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=800, width=800)
with right_column_4:
    st.write(
        "Infrastructure for bike is essential to promote more people to use bikes. The graph on the left shows roads expected to be most used by bikers in New York City. The darker the blue the heavier the traffic."
    )


st.write(
    "The most trafficted paths are actually those with the best infrastructure in town. The road on the west coast has great designated bike lanes while commuters on 23rd street make use of the bus lane to easily navigate throught the city - a place where improvement of the biking infrastructure could help."
)
left_column_last, right_column_last = st.beta_columns(2)
left_column_last.image(
    image_1, caption="Bike lane by the sea, NYC", use_column_width="always"
)
right_column_last.image(
    image_2, caption="Bus lane used by biker, NYC", use_column_width="always"
)

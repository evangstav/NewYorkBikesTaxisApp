import pickle

import pandas as pd
import plotly.express as px

# Run counts for subscriber
# with open("/dataset/Generated_data/costumer_dict.pickle", "rb") as handle:
# costumer_dict = pickle.load(handle)
# with open("/dataset/Generated_data/subscriber_dict.pickle", "rb") as handle:
# subscriber_dict = pickle.load(handle)
with open("datasets/taxi_dict.pickle", "rb") as handle:
    taxi_dict = pickle.load(handle)

data_taxi = pd.DataFrame(
    zip(list(range(24)), taxi_dict["month_count"]),
    columns=["months", "count"],
    index=None,
)

fig_taxi = px.bar(data_taxi, x="months", y="count")
fig_taxi.show()

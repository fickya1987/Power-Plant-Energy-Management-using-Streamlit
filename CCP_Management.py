import pandas as pd
import streamlit as st
import plotly.express as px
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Power Plant Energy Management")
st.header(" Indonesia Combined Cycle Power Plant Management")

st.write(""" 
            Features consist of hourly average:
            
            1. Ambient variables Temperature (T), 
            
            2. Ambient Pressure (AP),
             
            3. Relative Humidity (RH) and 
            
            4. Exhaust Vacuum (V) 
            
            To predict the net hourly Power Electrical output (PE) of the plant.
            """) 

# Load the data
df = pd.read_csv("Data_Regression.csv")

# Make the data appear as a dataframe on the webapp

st.subheader("**The CCPP Data in Tabular Form**")

data_frame = st.dataframe(df)


st.write("              ")


st.subheader("The plots below show the correlation between the individual features and the target output electrical power")

# Now show the individual correlation plots
# Plots will be shown as matplotlib figures.

fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(df["AT"], df["PE"])
axs[0, 0].set_title("AT vs PE")
axs[1, 0].scatter(df["V"], df["PE"])
axs[1, 0].set_title("V vs PE")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].scatter(df["AP"], df["PE"])
axs[0, 1].set_title("AP vs PE")
axs[1, 1].scatter(df["RH"], df["PE"])
axs[1, 1].set_title("RH vs PE")
fig.tight_layout()

st.write("          ")
st.pyplot(fig)

st.write("       ")
st.write(" The plots give us an idea how the features relate to the target.")
st.write("       ")
st.subheader(" An XGBoost model was trained on the data and it will be used to make inferences.")
st.write("       ")
st.write("**Please view the sidebar panel, to enter input features, that will be used to make predictions** ")
st.write("       ")


# Make the side bar panel
st.sidebar.header("User Input Features")
st.sidebar.markdown("                    ")
st.sidebar.markdown("Choose the values that will be used to make a prediction")

# Add Slider for users to pick values


def user_input_features():
    ambient_temperature = st.sidebar.slider("Ambient Temperature (AT): ", float(df.AT.min()), float(df.AT.max()), float(df.AT.mean()))
    vacuum = st.sidebar.slider("Exhaust Vacuum (EV): ", float(df.V.min()), float(df.V.max()), float(df.V.mean()))
    ambient_pressure = st.sidebar.slider("Ambient Pressure (AP): ", float(df.AP.min()), float(df.AP.max()), float(df.AP.mean()))
    relative_humidity = st.sidebar.slider("Relative Humidity (RH): ", float(df.RH.min()), float(df.RH.max()), float(df.RH.mean()))

    # Put the slider values that will be picked into a dictionary
    data = {
        "Ambient Temperature": ambient_temperature,
        "Exhaust Vacuum": vacuum,
        "Ambient Pressure": ambient_pressure,
        "Relative Humidity": relative_humidity
    }

    # Convert the dictionary to a data frame to store the picked values
    features = pd.DataFrame(data, index=[0])
    return features


input_features = user_input_features()

# Display the input features the user chose

st.write("                       ")
st.subheader("User Input Features")
st.write("                       ")
st.write(input_features)

st.write("                       ")
st.subheader("Prediction")
st.write("                       ")

# You can pick one model to make predictions.

xgboost_regressor = pickle.load(open('xgboost_regressor.pkl', 'rb'))
#random_forest_regressor = pickle.load(open('random_forest_regressor.pkl', 'rb'))


prediction = xgboost_regressor.predict(input_features)

prediction_value = float("{:.2f}".format(prediction[0]))

final_prediction = str(prediction_value) + " **kW**"

st.write(final_prediction)














import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="ValueScope | Housing Estimator", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; }
      .small-note { font-size: 0.9rem; opacity: 0.8; }
      .metric-card {
        padding: 1rem;
        border: 1px solid rgba(120,120,120,0.25);
        border-radius: 14px;
        background: rgba(240,240,240,0.2);
      }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")

model = joblib.load(MODEL_PATH)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Rooms_per_Occup"] = df["AveRooms"] / df["AveOccup"]
    df["Bedrooms_Ratio"] = df["AveBedrms"] / df["AveRooms"]
    df["Income_per_Room"] = df["MedInc"] / df["AveRooms"]
    return df

def predict_one(row: dict) -> float:
    df = pd.DataFrame([row])
    df = add_features(df)
    return float(model.predict(df)[0])

st.title("ValueScope")
st.write("A compact housing value estimator built from a supervised regression model (California Housing dataset).")

st.caption("Tip: Use the presets to quickly test different scenarios, then adjust inputs to explore sensitivity.")

presets = {
    "Custom (manual inputs)": None,
    "Coastal, higher income": {
        "MedInc": 8.3, "HouseAge": 35, "AveRooms": 6.9, "AveBedrms": 1.02,
        "Population": 900, "AveOccup": 2.6, "Latitude": 37.88, "Longitude": -122.23
    },
    "Inland, moderate income": {
        "MedInc": 3.5, "HouseAge": 20, "AveRooms": 5.2, "AveBedrms": 1.05,
        "Population": 1800, "AveOccup": 3.2, "Latitude": 35.37, "Longitude": -119.02
    },
    "Dense area, lower income": {
        "MedInc": 2.1, "HouseAge": 25, "AveRooms": 3.8, "AveBedrms": 1.10,
        "Population": 4500, "AveOccup": 4.0, "Latitude": 34.05, "Longitude": -118.24
    }
}

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Inputs")

    preset_name = st.selectbox("Scenario preset", list(presets.keys()))
    preset = presets[preset_name]

    default = preset if preset else {
        "MedInc": 3.0, "HouseAge": 20.0, "AveRooms": 5.0, "AveBedrms": 1.0,
        "Population": 1000.0, "AveOccup": 3.0, "Latitude": 37.88, "Longitude": -122.23
    }

    c1, c2 = st.columns(2)
    with c1:
        MedInc = st.number_input("Median income (MedInc)", min_value=0.0, value=float(default["MedInc"]), step=0.1)
        HouseAge = st.number_input("House age (HouseAge)", min_value=0.0, value=float(default["HouseAge"]), step=1.0)
        AveRooms = st.number_input("Average rooms (AveRooms)", min_value=0.1, value=float(default["AveRooms"]), step=0.1)
        AveBedrms = st.number_input("Average bedrooms (AveBedrms)", min_value=0.1, value=float(default["AveBedrms"]), step=0.1)
    with c2:
        Population = st.number_input("Population", min_value=0.0, value=float(default["Population"]), step=10.0)
        AveOccup = st.number_input("Average occupancy (AveOccup)", min_value=0.1, value=float(default["AveOccup"]), step=0.1)
        Latitude = st.number_input("Latitude", value=float(default["Latitude"]), step=0.01, format="%.2f")
        Longitude = st.number_input("Longitude", value=float(default["Longitude"]), step=0.01, format="%.2f")

    with st.expander("What do these inputs mean?"):
        st.markdown(
            """
            - **MedInc**: Median income indicator for the area (strongest predictor in EDA).
            - **HouseAge**: Average age of houses in the area.
            - **AveRooms / AveBedrms**: Average rooms/bedrooms per household.
            - **Population / AveOccup**: Density signals.
            - **Latitude / Longitude**: Captures location effects (coastal/inland patterns).
            """
        )

    run = st.button("Estimate value")

with right:
    st.subheader("Results")

    tab1, tab2 = st.tabs(["Prediction", "Sensitivity"])

    row = {
        "MedInc": float(MedInc),
        "HouseAge": float(HouseAge),
        "AveRooms": float(AveRooms),
        "AveBedrms": float(AveBedrms),
        "Population": float(Population),
        "AveOccup": float(AveOccup),
        "Latitude": float(Latitude),
        "Longitude": float(Longitude),
    }

    if run:
        pred = predict_one(row)

        with tab1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Predicted median house value (MedHouseVal)", f"{pred:.4f}")
            st.markdown(
                '<div class="small-note">Note: The dataset target is capped at 5.0, so predictions near the upper range can be affected by that limitation.</div>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            engineered = add_features(pd.DataFrame([row]))
            st.write("Engineered features used by the model:")
            st.dataframe(engineered[["Rooms_per_Occup", "Bedrooms_Ratio", "Income_per_Room"]])

        with tab2:
            st.write("Sensitivity check: how prediction changes when MedInc varies (other inputs fixed).")

            income_range = np.linspace(max(0.1, MedInc * 0.5), MedInc * 1.5, 20)
            preds = []
            for inc in income_range:
                tmp = row.copy()
                tmp["MedInc"] = float(inc)
                preds.append(predict_one(tmp))

            fig = plt.figure()
            plt.plot(income_range, preds)
            plt.xlabel("MedInc")
            plt.ylabel("Predicted MedHouseVal")
            plt.title("Prediction sensitivity to MedInc")
            st.pyplot(fig)

    else:
        st.info("Choose a preset or enter values, then click “Estimate value”.")


st.caption("Built by Malak Jawad for internship deliverable: a simple user interface to interact with the trained model.")
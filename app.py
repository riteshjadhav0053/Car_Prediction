import streamlit as st
import sys
import joblib
import os

MODEL_MAP_PATH = "models/make_model_map.pkl"

make_model_map = joblib.load(MODEL_MAP_PATH)

SPECS_PATH = "models/car_specs.pkl"

car_specs_dict = joblib.load(SPECS_PATH)

sys.path.append(".")

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🚗 Car Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the estimated price of a used car using a tuned XGBoost model.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CATEGORIES
# ============================================================

MAKES = [
    "Honda",
    "Maruti Suzuki",
    "Hyundai",
    "Toyota",
    "Mercedes-Benz",
    "BMW",
    "Skoda",
    "Nissan",
    "Renault",
    "Tata",
    "Volkswagen",
    "Ford",
    "Audi",
    "Mahindra",
    "MG",
    "Jeep",
    "Porsche",
    "Kia",
    "Land Rover",
    "Volvo",
    "Maserati",
    "Jaguar",
    "Isuzu",
    "Fiat",
    "MINI",
    "Ferrari",
    "Mitsubishi",
    "Datsun",
    "Lamborghini",
    "Chevrolet",
    "Ssangyong",
    "Rolls-Royce",
    "Lexus"
]


FUEL_TYPES = [
    "Petrol",
    "Diesel",
    "CNG",
    "LPG",
    "Electric",
    "CNG + CNG",
    "Hybrid",
    "Petrol + CNG",
    "Petrol + LPG"
]


TRANSMISSIONS = [
    "Manual",
    "Automatic"
]


LOCATIONS = [
    "Pune",
    "Ludhiana",
    "Lucknow",
    "Mangalore",
    "Mumbai",
    "Coimbatore",
    "Bangalore",
    "Delhi",
    "Raipur",
    "Kanpur",
    "Patna",
    "Vadodara",
    "Hyderabad",
    "Yamunanagar",
    "Gurgaon",
    "Jaipur",
    "Deoghar",
    "Agra",
    "Goa",
    "Warangal",
    "Jalandhar",
    "Noida",
    "Ahmedabad",
    "Mohali",
    "Navi Mumbai",
    "Ghaziabad",
    "Kolkata",
    "Zirakpur",
    "Nagpur",
    "Thane",
    "Faridabad",
    "Ranchi",
    "Chandigarh",
    "Amritsar",
    "Chennai",
    "Udupi",
    "Panvel",
    "Jamshedpur",
    "Aurangabad",
    "Rudrapur",
    "Nashik",
    "Varanasi",
    "Salem",
    "Dehradun",
    "Valsad",
    "Haldwani",
    "Dharwad",
    "Surat",
    "Indore",
    "Karnal",
    "Panchkula",
    "Mysore",
    "Rohtak",
    "Ambala Cantt",
    "Samastipur",
    "Unnao",
    "Purnea",
    "Bhubaneswar",
    "Kheda",
    "Kollam",
    "Meerut",
    "Ernakulam",
    "Kharar",
    "Mirzapur",
    "Bhopal",
    "Gorakhpur",
    "Guwahati",
    "Allahabad",
    "Muzaffurpur",
    "Faizabad",
    "Kota",
    "Pimpri-Chinchwad",
    "Dak. Kannada",
    "Ranga Reddy",
    "Bulandshahar",
    "Roorkee",
    "Siliguri"
]


COLORS = [
    "Grey",
    "White",
    "Maroon",
    "Red",
    "Blue",
    "Orange",
    "Silver",
    "Brown",
    "Black",
    "Bronze",
    "Gold",
    "Beige",
    "Green",
    "Yellow",
    "Purple",
    "Others",
    "Pink"
]


OWNERS = [
    "First",
    "Second",
    "Third",
    "Fourth",
    "UnRegistered Car",
    "4 or More"
]


SELLER_TYPES = [
    "Corporate",
    "Individual",
    "Commercial Registration"
]


DRIVETRAINS = [
    "FWD",
    "RWD",
    "AWD",
    "Unknown"
]


# ============================================================
# FORM
# ============================================================

with st.form("car_prediction_form"):

    st.subheader("🔧 Basic Car Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        make = st.selectbox(
            "Make",
            MAKES
        )

        available_models = make_model_map[make]

        model = st.selectbox(
            "Model",
            available_models
        )

        specs = car_specs_dict.get(
            (make, model),
            {}
        )


        year = st.number_input(
            "Manufacturing Year",
            min_value=1980,
            max_value=2026,
            value=2018,
            step=1
        )

        kilometer = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=2_000_000,
            value=50_000,
            step=1
        )

        car_age = 2026 - year
        
        if car_age <= 0:
            car_age = 1

        km_per_year = kilometer / car_age


    with col2:

        fuel_type = st.selectbox(
            "Fuel Type",
            FUEL_TYPES
        )

        transmission = st.selectbox(
            "Transmission",
            TRANSMISSIONS
        )

        location = st.selectbox(
            "Location",
            LOCATIONS
        )

        color = st.selectbox(
            "Color",
            COLORS
        )

    with col3:

        owner = st.selectbox(
            "Owner",
            OWNERS
        )

        seller_type = st.selectbox(
            "Seller Type",
            SELLER_TYPES
        )

        drivetrain = st.selectbox(
            "Drivetrain",
            DRIVETRAINS
        )

    st.divider()

    st.subheader("⚙️ Engine & Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        engine = specs.get("Engine", 0.0)

        max_power = specs.get("Max Power", 0.0)

        max_torque = specs.get("Max Torque", 0.0)

        st.number_input(
            "Engine",
            value=float(engine),
            disabled=True
        )

        st.number_input(
            "Max Power",
            value=float(max_power),
            disabled=True
        )

        st.number_input(
            "Max Torque",
            value=float(max_torque),
            disabled=True
        )

    with col2:

        length = specs.get("Length", 0.0)

        width = specs.get("Width", 0.0)

        height = specs.get("Height", 0.0)

        st.number_input(
            "Length",
            value=float(length),
            disabled=True
        )

        st.number_input(
            "Width",
            value=float(width),
            disabled=True
        )

        st.number_input(
            "Height",
            value=float(height),
            disabled=True
        )

    with col3:

        seating_capacity = specs.get(
            "Seating Capacity",
            0.0
        )

        fuel_tank_capacity = specs.get(
            "Fuel Tank Capacity",
            0.0
        )

        st.number_input(
            "Seating Capacity",
            value=float(seating_capacity),
            disabled=True
        )

        st.number_input(
            "Fuel Tank Capacity",
            value=float(fuel_tank_capacity),
            disabled=True
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔮 Predict Car Price",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    if not model:
        st.warning("⚠️ Please enter the car model.")

    elif not engine:
        st.warning("⚠️ Please enter the engine specification.")

    elif not max_power:
        st.warning("⚠️ Please enter the maximum power.")

    elif not max_torque:
        st.warning("⚠️ Please enter the maximum torque.")

    else:

        try:

            data = CustomData(
                make=make,
                model=model,
                year=year,
                kilometer=kilometer,
                fuel_type=fuel_type,
                transmission=transmission,
                location=location,
                color=color,
                owner=owner,
                seller_type=seller_type,
                engine=engine,
                max_power=max_power,
                max_torque=max_torque,
                drivetrain=drivetrain,
                length=length,
                width=width,
                height=height,
                seating_capacity=seating_capacity,
                fuel_tank_capacity=fuel_tank_capacity
            )

            prediction_data = data.get_data_as_dataframe()

            pipeline = PredictPipeline()

            prediction = pipeline.predict(
                prediction_data
            )

            price = prediction[0]

            st.success("✅ Prediction completed successfully!")

            st.subheader("💰 Estimated Market Price")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Estimated Price",
                    f"₹ {price:,.0f}"
                )

            with col2:
                st.metric(
                    "Car Age",
                    f"{car_age} years"
                )

            with col3:
                st.metric(
                    "Usage",
                    f"{km_per_year:,.0f} km/year"
                )

        except Exception as e:

            st.error(
                "❌ Prediction failed. Please check your input values."
            )

            st.exception(e)
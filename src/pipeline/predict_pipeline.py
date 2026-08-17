import os
import joblib
import pandas as pd


class CustomData:

    def __init__(
        self,
        make,
        model,
        year,
        kilometer,
        fuel_type,
        transmission,
        location,
        color,
        owner,
        seller_type,
        engine,
        max_power,
        max_torque,
        drivetrain,
        length,
        width,
        height,
        seating_capacity,
        fuel_tank_capacity
    ):

        self.make = make
        self.model = model
        self.year = year
        self.kilometer = kilometer
        self.fuel_type = fuel_type
        self.transmission = transmission
        self.location = location
        self.color = color
        self.owner = owner
        self.seller_type = seller_type
        self.engine = engine
        self.max_power = max_power
        self.max_torque = max_torque
        self.drivetrain = drivetrain
        self.length = length
        self.width = width
        self.height = height
        self.seating_capacity = seating_capacity
        self.fuel_tank_capacity = fuel_tank_capacity

    def get_data_as_dataframe(self):

        reference_year = 2026

        car_age = reference_year - self.year

        # Avoid division by zero for a 2026 model
        if car_age <= 0:
            car_age = 1

        km_per_year = self.kilometer / car_age

        custom_data_input = {
            "Make": [self.make],
            "Model": [self.model],
            "Year": [self.year],
            "Kilometer": [self.kilometer],
            "Fuel Type": [self.fuel_type],
            "Transmission": [self.transmission],
            "Location": [self.location],
            "Color": [self.color],
            "Owner": [self.owner],
            "Seller Type": [self.seller_type],
            "Engine": [self.engine],
            "Max Power": [self.max_power],
            "Max Torque": [self.max_torque],
            "Drivetrain": [self.drivetrain],
            "Length": [self.length],
            "Width": [self.width],
            "Height": [self.height],
            "Seating Capacity": [self.seating_capacity],
            "Fuel Tank Capacity": [self.fuel_tank_capacity],
            "Car_age": [car_age],
            "km_per_year": [km_per_year]
        }

        return pd.DataFrame(custom_data_input)


class PredictPipeline:

    def __init__(self):

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../.."
            )
        )

        model_path = os.path.join(
            project_root,
            "models",
            "best_xgb_model.pkl"
        )

        preprocessor_path = os.path.join(
            project_root,
            "models",
            "preprocessor.pkl"
        )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, features):

        data = self.preprocessor.transform(features)

        prediction = self.model.predict(data)

        return prediction
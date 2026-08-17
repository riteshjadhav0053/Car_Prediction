from src.pipeline.predict_pipeline import CustomData, PredictPipeline


if __name__ == "__main__":

    # Sample car
    data = CustomData(
        make="Honda",
        model="Amaze 1.2 VX i-VTEC",
        year=2017,
        kilometer=50000,
        fuel_type="Petrol",
        transmission="Manual",
        location="Pune",
        color="Grey",
        owner="First",
        seller_type="Individual",
        engine=1198.0,
        max_power=87.0,
        max_torque=109.0,
        drivetrain="FWD",
        length=3990.0,
        width=1680.0,
        height=1505.0,
        seating_capacity=5.0,
        fuel_tank_capacity=35.0
    )

    # Convert input into DataFrame
    features = data.get_data_as_dataframe()

    print("\nInput Data:")
    print(features)

    # Load model and make prediction
    pipeline = PredictPipeline()

    prediction = pipeline.predict(features)

    print("\n" + "=" * 50)
    print(f"Predicted Car Price: ₹{prediction[0]:,.2f}")
    print("=" * 50)
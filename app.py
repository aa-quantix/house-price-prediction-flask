from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

# Keep this order identical to the columns used when the model was trained.
FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "sale_year",
    "sale_month",
]

# These inputs represent whole-number counts, categories, years, or ZIP codes.
INTEGER_FEATURES = {
    "bedrooms",
    "sqft_living",
    "sqft_lot",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "sqft_living15",
    "sqft_lot15",
    "sale_year",
    "sale_month",
}

FIELD_LABELS = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "sqft_living": "Living Area (sqft)",
    "sqft_lot": "Lot Area (sqft)",
    "floors": "Floors",
    "waterfront": "Waterfront (0 = no, 1 = yes)",
    "view": "View (0-4)",
    "condition": "Condition (1-5)",
    "grade": "Grade (1-13)",
    "sqft_above": "Above Ground Area (sqft)",
    "sqft_basement": "Basement Area (sqft)",
    "yr_built": "Year Built",
    "yr_renovated": "Year Renovated (0 if never)",
    "zipcode": "Zipcode",
    "lat": "Latitude",
    "long": "Longitude",
    "sqft_living15": "Living Area of Nearby Houses (sqft)",
    "sqft_lot15": "Lot Area of Nearby Houses (sqft)",
    "sale_year": "Sale Year",
    "sale_month": "Sale Month",
}

# Supply useful browser-side constraints while retaining server-side validation.
FIELDS = [
    {
        "name": feature,
        "label": FIELD_LABELS[feature],
        "step": "1" if feature in INTEGER_FEATURES else "any",
        "min": "1" if feature in {"bedrooms", "floors"} else "0"
        if feature not in {"lat", "long"}
        else None,
        "max": {
            "waterfront": "1",
            "view": "4",
            "condition": "5",
            "grade": "13",
            "sale_month": "12",
            "lat": "90",
            "long": "180",
        }.get(feature),
    }
    for feature in FEATURES
]

MODEL_PATH = Path(__file__).resolve().parent / "final_house_price_model.pkl"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Place final_house_price_model.pkl "
        "in the same folder as app.py."
    )

# Load the already-trained model once when the Flask application starts.
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        # Keep submitted values in the form if validation reports a problem.
        form_data = {
            feature: request.form.get(feature, "").strip() for feature in FEATURES
        }

        try:
            values = {}
            for feature in FEATURES:
                raw_value = form_data[feature]
                label = FIELD_LABELS[feature]

                if not raw_value:
                    raise ValueError(f"Please enter a value for {label.lower()}.")

                try:
                    numeric_value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(f"{label} must be a valid number.") from exc

                if not np.isfinite(numeric_value):
                    raise ValueError(f"{label} must be a finite number.")

                if feature in INTEGER_FEATURES:
                    if not numeric_value.is_integer():
                        raise ValueError(f"{label} must be a whole number.")
                    values[feature] = int(numeric_value)
                else:
                    values[feature] = numeric_value

            # Catch common out-of-range values before calling the model.
            if values["waterfront"] not in (0, 1):
                raise ValueError("Waterfront must be 0 (no) or 1 (yes).")
            if not 0 <= values["view"] <= 4:
                raise ValueError("View must be between 0 and 4.")
            if not 1 <= values["condition"] <= 5:
                raise ValueError("Condition must be between 1 and 5.")
            if not 1 <= values["grade"] <= 13:
                raise ValueError("Grade must be between 1 and 13.")
            if not 1 <= values["sale_month"] <= 12:
                raise ValueError("Sale month must be between 1 and 12.")
            if any(values[name] < 0 for name in (
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "sqft_lot",
                "floors",
                "sqft_above",
                "sqft_basement",
                "sqft_living15",
                "sqft_lot15",
            )):
                raise ValueError("Bedrooms, floors, and area values cannot be negative.")
            if not -90 <= values["lat"] <= 90:
                raise ValueError("Latitude must be between -90 and 90.")
            if not -180 <= values["long"] <= 180:
                raise ValueError("Longitude must be between -180 and 180.")

            # Explicit columns keep both feature names and their training order.
            input_data = pd.DataFrame([values], columns=FEATURES)
            predicted_price = float(model.predict(input_data)[0])
            if not np.isfinite(predicted_price):
                raise ValueError("The model returned an invalid prediction.")

            prediction = f"${predicted_price:,.2f}"
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        fields=FIELDS,
        form_data=form_data,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    # Debug mode is convenient for local student projects; do not use it publicly.
    app.run(debug=True)

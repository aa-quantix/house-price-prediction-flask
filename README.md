# House Price Prediction

A small Flask web application that estimates a house price using a previously trained `HistGradientBoostingRegressor`. The app loads the saved model and does not retrain or modify it.

## Project Files

```text
house_price_flask/
├── app.py
├── final_house_price_model.pkl
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Run in VS Code on Windows

Open this folder in VS Code, open a terminal in the project directory, and run:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> in a browser. To stop the local server, press `Ctrl+C` in the terminal.

## Model and Inputs

The bundled model is expected to be compatible with scikit-learn 1.6.1, which is pinned in `requirements.txt`. The app passes a pandas DataFrame containing these features in the exact training order:

```text
bedrooms, bathrooms, sqft_living, sqft_lot, floors,
waterfront, view, condition, grade, sqft_above,
sqft_basement, yr_built, yr_renovated, zipcode, lat,
long, sqft_living15, sqft_lot15, sale_year, sale_month
```

All fields are required. Integer-valued fields must contain whole numbers. The app checks common ranges such as waterfront (`0` or `1`), view (`0` through `4`), condition (`1` through `5`), grade (`1` through `13`), and sale month (`1` through `12`). Predictions are shown in currency format. Estimates are for educational use and are not an appraisal.

## How It Works

```text
User enters house data
        ↓
HTML form
        ↓
Flask POST request
        ↓
Pandas DataFrame with training feature order
        ↓
Loaded model predicts a price
        ↓
Formatted result displayed in the browser
```

## Troubleshooting

- **`ModuleNotFoundError`**: Make sure the virtual environment is activated, then run `pip install -r requirements.txt` from this folder. In VS Code, select the `venv` interpreter.
- **Model file not found**: Confirm `final_house_price_model.pkl` is in the same folder as `app.py`.
- **Port 5000 is already in use**: Stop the other local server or change the port in the `app.run(...)` line in `app.py` and open the matching URL.
- **scikit-learn version warning or error**: Install the pinned version with `pip install scikit-learn==1.6.1` inside the activated environment. Do not re-save the model using another version as a workaround.
- **Invalid form input**: Fill every field, use whole numbers for integer-valued inputs, and check category and coordinate ranges in the error message.

## Development Note

The app runs with Flask's local development server and debug mode. Do not expose the debug server to the public internet.
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
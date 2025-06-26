import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib
import numpy as np

# 1. Load pre-trained scikit-learn model
model = joblib.load("regression_model.pkl")

# 2. Initialize Dash app
app = dash.Dash(__name__)

# 3. Define layout with input slider & display area
app.layout = html.Div([
    html.H2("House Price Predictor"),
    dcc.Slider(
        id="sqft-slider",
        min=200, max=5000, step=50,
        value=1000,
        marks={i: f"{i}" for i in range(500, 5001, 500)}
    ),
    html.Div(id="prediction-output", style={"marginTop": 20})
])

# 4. Callback: update prediction when slider moves
@app.callback(
    Output("prediction-output", "children"),
    [Input("sqft-slider", "value")]
)
def update_price(sqft):
    # 4.1 Predict price given square footage
    pred = model.predict(np.array([[sqft]]))[0]
    # 4.2 Return formatted text
    return f"Estimated Price: ${pred:,.2f}"

# 5. Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
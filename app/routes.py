from flask import current_app as app, render_template
from core.demand_forecast import DemandForecast

demand_forecast = DemandForecast()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.jinja2", title="Show Task")

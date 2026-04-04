from flask import Flask, render_template, request
from predict import predict_default

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        loan_amt_outstanding = float(request.form["loan_amt_outstanding"])
        income = float(request.form["income"])
        years_employed = float(request.form["years_employed"])
        fico_score = float(request.form["fico_score"])

        result = predict_default(
            loan_amt_outstanding=loan_amt_outstanding,
            income=income,
            years_employed=years_employed,
            fico_score=fico_score,
        )

        probability_percent = round(result["probability_default"] * 100, 2)
        threshold_percent = round(result["threshold"] * 100, 2)

        if result["prediction"] == 1:
            decision_label = "Risque de défaut élevé"
            decision_class = "risk"
        else:
            decision_label = "Risque de défaut faible"
            decision_class = "safe"

        return render_template(
            "result.html",
            prediction=result["prediction"],
            probability_percent=probability_percent,
            threshold_percent=threshold_percent,
            decision_label=decision_label,
            decision_class=decision_class,
            loan_amt_outstanding=loan_amt_outstanding,
            income=income,
            years_employed=years_employed,
            fico_score=fico_score,
        )

    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
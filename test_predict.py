from predict import predict_default

result = predict_default(
    loan_amt_outstanding=5000,
    income=30000,
    years_employed=5,
    fico_score=650,
)

print(result)
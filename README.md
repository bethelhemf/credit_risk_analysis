# Credit Scoring Business Understanding

## 1. Basel II and the Importance of Interpretable Credit Risk Models

The Basel II Accord places strong emphasis on the accurate measurement, validation, and management of credit risk. Financial institutions must be able to demonstrate that their risk models are reliable, transparent, and supported by proper documentation. Consequently, a credit scoring model should not only predict risk accurately but also explain how lending decisions are reached.

Interpretable models such as Logistic Regression with Weight of Evidence (WoE) transformations are commonly used because they provide clear relationships between borrower characteristics and the probability of default. This transparency allows regulators, auditors, risk managers, and business stakeholders to understand and validate model behavior. Well-documented models also support regulatory compliance, model governance, monitoring, and periodic validation.

From a Basel II perspective, model explainability is critical because lending decisions directly affect capital allocation, risk management practices, and regulatory reporting. A highly accurate model that cannot be adequately explained may create compliance and governance challenges.

## 2. Why a Proxy Variable Is Necessary and the Risks It Introduces

Credit scoring models are typically supervised learning models that require a target variable indicating whether a borrower has defaulted. However, many datasets do not contain a direct "default" label. In such situations, a proxy variable must be created to represent default-like behavior.

Examples of proxy indicators include:

- Extended payment delinquency (e.g., 90+ days past due)
- Loan write-offs
- Persistent arrears
- Serious repayment difficulties

The proxy serves as a substitute target variable that allows the model to learn patterns associated with credit risk.

However, proxy-based prediction introduces several business risks:

- **Label Noise:** The proxy may not perfectly represent actual default behavior.
- **Misclassification Risk:** Some customers classified as risky may ultimately repay, while some classified as safe may later default.
- **Bias Risk:** An improperly designed proxy may disproportionately affect certain customer groups.
- **Model Drift:** The relationship between the proxy and true default behavior may change over time.
- **Regulatory and Governance Risk:** Decisions based on an imperfect proxy can be more difficult to justify to regulators and auditors.

Therefore, proxy variables should be carefully designed, validated with business experts, and continuously monitored to ensure they remain representative of actual credit risk outcomes.

## 3. Trade-Offs Between Logistic Regression with WoE and Gradient Boosting

In regulated financial environments, model selection involves balancing predictive performance against interpretability and regulatory compliance.

| Aspect | Logistic Regression with WoE | Gradient Boosting |
|----------|-----------------------------|------------------|
| Interpretability | High | Low |
| Regulatory Acceptance | Strong | Moderate; requires additional explainability |
| Ease of Validation | High | More complex |
| Documentation Requirements | Relatively simple | Extensive |
| Predictive Accuracy | Good | Often higher |
| Ability to Capture Nonlinear Relationships | Limited | Strong |
| Ability to Capture Feature Interactions | Limited | Strong |
| Business Explainability | Easy to communicate | More difficult |
| Model Governance | Simpler | More demanding |

Logistic Regression with WoE remains a standard approach in credit scoring because its decisions can be easily interpreted and justified. Stakeholders can clearly understand how each variable contributes to risk, making regulatory approval and model validation more straightforward.

Gradient Boosting models often achieve superior predictive performance because they can capture complex nonlinear relationships and interactions among variables. However, they are less transparent and typically require additional explainability techniques, such as SHAP values, to satisfy regulatory and governance requirements.

As a result, financial institutions must balance predictive power with transparency. While advanced machine learning models may improve risk prediction, interpretable models often remain preferable in highly regulated credit decisioning environments due to their explainability, auditability, and regulatory acceptance.

## References

1. Basel Committee on Banking Supervision. *International Convergence of Capital Measurement and Capital Standards: A Revised Framework (Basel II)*.
2. Academia Sinica. *Credit Scoring Statistical Analysis*.
3. Hong Kong Monetary Authority (HKMA). *Alternative Credit Scoring*.
4. World Bank. *Credit Scoring Approaches Guidelines*.
5. Siddiqi, N. *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*.
6. Corporate Finance Institute (CFI). *Credit Risk*.

[View EDA Notebook (External Link)](https://nbviewer.org/github/bethelhemf/credit_risk_analysis/blob/main/notebooks/eda.ipynb)
## Task 2: Exploratory Data Analysis Results

The full EDA can be viewed here: [Online Notebook Viewer (Recommended)](https://nbviewer.org/github/bethelhemf/credit_risk_analysis/blob/main/notebooks/eda.ipynb)

### Key Insights:
1. **Monetary Skewness:** Transactions have extreme outliers ().
2. **Risk Correlation:** Transaction Value has a 0.57 correlation with FraudResult.
3. **Category Dominance:** Financial Services and Airtime are the primary transaction types.


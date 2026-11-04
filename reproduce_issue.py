
import pandas as pd
from feature_engine.datetime import DatetimeSubtraction

X = pd.DataFrame({
    "end": ["2026-01-02"],
    "start": ["2026-01-01"]
})

transformer = DatetimeSubtraction(
    variables=["end"],
    reference=["start"],
    drop_original=True
)

result = transformer.fit_transform(X)

print("Actual columns:", result.columns.tolist())
print("Reported feature names:", transformer.get_feature_names_out())
print("Values:", result.to_numpy().tolist())

print("\nTesting pandas output")
transformer.set_output(transform="pandas").transform(X)

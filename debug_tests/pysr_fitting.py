from src.comsolComponent import ComsolComponent
from pysr import PySRRegressor


def symbolic_regression(X, y):
    model = PySRRegressor()
    model.fit(X, y)
    return model


df = df[df["Time_s"] == df["Time_s"].max()]
y = df["Current (A)"]
df = df.rename(columns={"S": "ion_pair_rate"})
X = df[
    [
        "ion_pair_rate",
        "Udc_kV",
        "electric_field_norm_max",
    ]
]

model = symbolic_regression(X, y)

plt.scatter(y, model.predict(X, 3))
plt.show()

from sklearn.linear_model import LinearRegression
import pandas

df = pandas.read_csv("data.csv")
train_df_x = df[["median age", "median income"]].iloc[:7]
train_df_y = df[["private health"]].iloc[:7]


test_list = df.iloc[7:].values
reg = LinearRegression()
reg.fit(train_df_x.values, train_df_y.values)

test_data = {
    "Median age":list(test_list[i][0] for i in range(3)),
    "Median income":list(test_list[i][0] for i in range(3)),
    "Private health":[],
    "Expected private health":list(test_list[i][2] for i in range(3)),
    "Error":[]
}

for i in range(3):
    prediction =reg.predict([test_list[i][:2]])[0][0]
    test_data["Private health"].append(prediction)
    test_data["Error"].append(prediction - test_list[i][2])

test_df = pandas.DataFrame(test_data)

print("Training Data")
print(df.iloc[:7])
print("\n1. Coefficients of X matrix and Y ~",*reg.coef_)
print("\n2. Test Data")
print(test_df)

print("\n3. Custom prediction for '41' years age and '52' thousand income -",end=" ")
print(reg.predict([[41,52]])[0][0])

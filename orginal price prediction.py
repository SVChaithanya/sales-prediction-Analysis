import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\surya\ML_Project\sales_data.csv.csv")
df['orginal price'] = df['price']-(df['price']/100*df['discount'])
print(df)
customer_price = df.groupby('customer_region')['orginal price'].sum()
category_price=df.groupby('category')['orginal price'].sum()
product_price=df.groupby('product')['orginal price'].sum()

fig=plt.subplot()
customer_price.plot(kind='bar')
plt.title('customer v/s orginal price')
plt.show()

category_price.plot(kind='barh')
plt.title('category v/s orginal price')
plt.show()

product_price.plot(kind='line')
plt.title('product v/s orginal price')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

X =df[['product','customer_region','price','discount']]
y=df['orginal price']
num = ['price','discount']
cat = ['product','customer_region']
process = ColumnTransformer(transformers=[('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = Pipeline([('process',process),('model',RandomForestRegressor(n_estimators=100,max_depth=50,random_state=42))])

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(r2_score(y_test,y_pred))
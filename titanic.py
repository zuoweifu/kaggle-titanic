#import libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd

#import data
X = pd.read_csv("train.csv")
y = X.pop("Survived")

#print out initial data
X.describe()

#claculate mean of age
X["Age"].fillna(X.Age.mean(),inplace=True)
X.describe()

#rule out all non-numerical variables
numerica_varialbles= list(X.dtypes[X.dtypes != object].index)
X[numerica_varialbles].head()


model = RandomForestRegressor(n_estimators=100,oob_score=True, random_state=42)
model.fit(X[numerica_varialbles],y)
model.oob_score_
y_oob = model.oob_prediction_
print "c-stat: ",roc_auc_score(y,y_oob)
'''
the result is 0.73995515504
'''


def describe_categorical(X):
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))

describe_categorical(X)

'''
througn observation, i found that name, ticket and PassengerId might not be useful for the model
so just droped them
'''
X.drop(["Name","Ticket","PassengerId"],axis=1,inplace=True)

'''
only keep the first cahr of cabin
'''
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"

X["Cabin"] = X.Cabin.apply(clean_cabin)

X.Cabin

categorical_variables = ['Sex','Cabin','Embarked']
for variable in categorical_variables:
    X[variable].fillna("Missing",inplace=True)
    dummies = pd.get_dummies(X[variable],prefix=variable)
    X = pd.concat([X,dummies],axis=1)
    X.drop([variable],axis=1,inplace=True)

def printall(X,max_rows=10):
    from IPython.display import display,HTML
    display(HTML(X.to_html(max_rows=max_rows)))

printall(X)

model = RandomForestRegressor(100,oob_score=True, n_jobs=-1,random_state=42)
model.fit(X,y)
print "c-stat: ", roc_auc_score(y,model.oob_prediction_)
'''
the result is 0.863521128261
'''
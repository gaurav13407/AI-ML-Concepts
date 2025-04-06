## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('Travel.csv')
df.head()

## HAndling missing Value
## Checking all the columns
df.isnull().sum()

## Checking all the categories
df['Gender'].value_counts()

df['MaritalStatus'].value_counts()

df['TypeofContact'].value_counts()

df['Gender']=df['Gender'].replace('Fe Male','Female')
df['MaritalStatus']=df['MaritalStatus'].replace('Single','Unmarried')

df.head()

feature_with_na = [feature for feature in df.columns if df[feature].isnull().sum() >= 1]

# Displaying missing value percentages
for feature in feature_with_na:
    print(f"{feature}: {np.round(df[feature].isnull().mean() * 100, 5)}% missing values")

## Statistic on numerical cloumns (Null cols)
df[feature_with_na].select_dtypes(exclude='object').describe()

df.Age.fillna(df.Age.median(),inplace=True)
## Type of Contract
df.TypeofContact.fillna(df.TypeofContact.mode()[0],inplace=True)
## Duration Of pitch
df.DurationOfPitch.fillna(df.DurationOfPitch.median(),inplace=True)
##Number of FOllowups
df.NumberOfFollowups.fillna(df.NumberOfFollowups.mode().mode()[0],inplace=True)
##PrefereedPropertyStar
df.PreferredPropertyStar.fillna(df.PreferredPropertyStar.mode()[0],inplace=True)
## NUmber of tips
df.NumberOfTrips.fillna(df.NumberOfTrips.median(),inplace=True)
## Number of Chlidren Visting
df.NumberOfChildrenVisiting.fillna(df.NumberOfChildrenVisiting.mode()[0],inplace=True)
## Montly Income
df.MonthlyIncome.fillna(df.MonthlyIncome.median(),inplace=True)

df.head()


df.drop('CustomerID',inplace=True,axis=1)

df.head()

## Create a new column for feature
df['TotalVisting']=df['NumberOfChildrenVisiting']+df['NumberOfPersonVisiting']
df.drop(columns=['NumberOfChildrenVisiting','NumberOfPersonVisiting'],axis=1,inplace=True)

## Get All the Numeric Feature
num_feature=[feature for feature in df.columns if df[feature].dtype!='O']
print('Num Of numerical Feature:',len(num_feature))

## Categorical Feature
cat_feature=[feature for feature in df.columns if df[feature].dtype=='O']
print('Num Of categorical Feature:',len(cat_feature))

## Discrete Feature
dis_feature=[feature for feature in df.columns if len(df[feature].unique())<=25]
print('Num Of discrete Feature:',len(dis_feature))

## Continous Feature
con_feature=[feature for feature in num_feature if feature not in dis_feature]
print('Num Of continous Feature:',len(con_feature))

from sklearn.model_selection import train_test_split
X=df.drop(['ProdTaken'],axis=1)
y=df['ProdTaken']

X.head()

y.value_counts()

X.head()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

cat_feature = X.select_dtypes(include="object").columns
num_feature = X.select_dtypes(exclude="object").columns

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_feature),
        ("StandardScaler", numeric_transformer, num_feature)  # ✅ Correct tuple
    ]
)


preprocessor

X_train=preprocessor.fit_transform(X_train)

pd.DataFrame(X_train)

X_test=preprocessor.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score,roc_curve

models={
    "Logistic Regression":LogisticRegression(),
    "RandomForest":RandomForestClassifier(),
    "Decision Tree":DecisionTreeClassifier(),
    "GradientBoost":GradientBoostingClassifier(),
    "AdaBoostClassifier":AdaBoostClassifier()
}
for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)
    
    ## Make Prediction
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)
    
    ## Training Set Performance
    model_train_accuracy=accuracy_score(y_train,y_train_pred)
    model_train_f1=f1_score(y_train,y_train_pred,average='weighted')
    model_train_precision=precision_score(y_train,y_train_pred)
    model_train_recall=recall_score(y_train,y_train_pred)
    model_train_roauc_score=roc_auc_score(y_train,y_train_pred)
    
    ## Test set performance
    model_test_accuracy=accuracy_score(y_test,y_test_pred)
    model_test_f1=f1_score(y_test,y_test_pred,average='weighted')
    model_test_precision=precision_score(y_test,y_test_pred)
    model_test_recall=recall_score(y_test,y_test_pred)
    model_test_roauc_score=roc_auc_score(y_test,y_test_pred)
    print(list(models.keys())[i])
    
    print('Model performance on Training Set')
    print('-Acuuracy:{:.4f}'.format(model_train_accuracy))
    print('-F1 score:{:.4f}'.format(model_train_f1))
    
    print('-Precision:{:.4f}'.format(model_train_precision))
    print('-Recall:{:.4f}'.format(model_train_recall))
    print('- ROC Auc Score:{:.4f}'.format(model_train_roauc_score))
    
    print('-------------------------------------------------------')
    print('model Performance on test set')
    print('-Acuuracy:{:.4f}'.format(model_test_accuracy))
    print('-F1 score:{:.4f}'.format(model_test_f1))
    
    print('-Precision:{:.4f}'.format(model_test_precision))
    print('-Recall:{:.4f}'.format(model_test_recall))
    print('- ROC Auc Score:{:.4f}'.format(model_test_roauc_score))
    print('-------------------------------------------------------')
    print('-------------------------------------------------------')

## Hyperparameter Tunning
rf_params={'max_depth':[5,8,15,None,10],
           'max_features':[5,7,"auto",8],
           'min_samples_split':[2,8,15,20],
           'n_estimators':[100,200,500,1000]}


gradient_paramas={"loss":['log_loss', 'exponential'],
                  "criterion":['friedman_mse', 'squared_error'],
                  "min_samples_split":[2,8,15,20],
                  "n_estimators":[100,200,500,1000],
                  "max_depth":[5,8,15,None,10]
                  }

gradient_paramas

## Model list for Hyperparameter Tuning
radomcv_model=[
    ("Rf",RandomForestClassifier(),rf_params),
    ("GB",GradientBoostingClassifier(),gradient_paramas)
]

from sklearn.model_selection import RandomizedSearchCV

model_param={}
for name,model,params in radomcv_model:
    random=RandomizedSearchCV(estimator=model,
                              param_distributions=params,
                              n_iter=100,
                              cv=3,
                              verbose=2,
                              n_jobs=-1)
    random.fit(X_train,y_train)
    model_param[name]=random.best_params_
    
for model_name in model_param:
    print(f"---------------- Best Params for {model_name}--------")
    print(model_param[model_name])

models={
    "RandomForest":RandomForestClassifier(n_estimators=100,min_samples_split=2,max_features=8,max_depth=15),
    "GradientBoostClasifier":GradientBoostingClassifier(n_estimators=1000,min_samples_split=20,max_depth=15,loss='log_loss',criterion=
                                                        'squared_error')
}
for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)
    
    ## Make Prediction
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)
    
    ## Training Set Performance
model_train_accuracy=accuracy_score(y_train,y_train_pred)
model_train_f1=f1_score(y_train,y_train_pred,average='weighted')
model_train_precision=precision_score(y_train,y_train_pred)
model_train_recall=recall_score(y_train,y_train_pred)
model_train_roauc_score=roc_auc_score(y_train,y_train_pred)
    
    ## Test set performance
model_test_accuracy=accuracy_score(y_test,y_test_pred)
model_test_f1=f1_score(y_test,y_test_pred,average='weighted')
model_test_precision=precision_score(y_test,y_test_pred)
model_test_recall=recall_score(y_test,y_test_pred)
model_test_roauc_score=roc_auc_score(y_test,y_test_pred)
print(list(models.keys())[i])
    
print('Model performance on Training Set')
print('-Acuuracy:{:.4f}'.format(model_train_accuracy))
print('-F1 score:{:.4f}'.format(model_train_f1))
    
print('-Precision:{:.4f}'.format(model_train_precision))
print('-Recall:{:.4f}'.format(model_train_recall))
print('- ROC Auc Score:{:.4f}'.format(model_train_roauc_score))
    
print('-------------------------------------------------------')
print('model Performance on test set')
print('-Acuuracy:{:.4f}'.format(model_test_accuracy))
print('-F1 score:{:.4f}'.format(model_test_f1))
    
print('-Precision:{:.4f}'.format(model_test_precision))
print('-Recall:{:.4f}'.format(model_test_recall))
print('- ROC Auc Score:{:.4f}'.format(model_test_roauc_score))
print('-------------------------------------------------------')
print('-------------------------------------------------------')


## Plot ROC AUC Curve
from sklearn.metrics import roc_auc_score,roc_curve
plt.figure()
## Add the models to the list that youn want to view on the ROC plot
auc_model=[
    {
        'label':'Gradient',
        'model':GradientBoostingClassifier(n_estimators=1000,min_samples_split=20,max_depth=15,loss='log_loss',criterion=
                                                        'squared_error'),
        'auc':0.8954
    }
]
## Create a loop through all the model
for algo in auc_model:
    model=algo['model']
    model.fit(X_train,y_train)
## Compute False positive and True positive Rate
    fpr,tpr,threshold=roc_curve(y_test,model.predict_proba(X_test)[:,1])
    ## Calculate the area under the curve
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # Compute AUC score
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (algo['label'], auc))  # ✅ Pass AUC

## Customize Setting
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('1-specificity(Fasle positive rate)')
plt.ylabel('Sensitive (True Podtive Rate)')
plt.title('Reciever Operating Characterstics')
plt.legend(loc="lower right")
plt.savefig("auc.png")
plt.show()


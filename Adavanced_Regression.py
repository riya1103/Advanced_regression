import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
df.count
df=pd.read_csv("train.csv")
df1=pd.read_csv('test.csv')
y=df['SalePrice']

df=df.append(df1)

df.info
describe=df.describe()
df=df.drop(columns=['Id'])
X=df
X=X.drop(columns=['SalePrice'])

X=X.drop(columns=['SalePrice'])
#calculating unique values of all the columns at once 
unique=df.nunique()

df.skew(axis=0,skipna=True)

df.skew(axis=1,skipna=True)

#Applying log1p transformation to remove the skewness
df['SalePrice'].hist(bins=10)
df['SalePrice']=np.log1p(df['SalePrice'])
df['SalePrice'].hist(bins=10)

df.columns.to_series().groupby(df.dtypes).groups


df['MSSubClass']=df['MSSubClass'].astype(str)
df['MoSold']=df['MoSold'].astype(str)

df.info()


X.info()
X.isnull().any().sum()

#IMPORTANT!!
a=pd.DataFrame()
a=X.isna().sum()

X['YrSold'].plot.bar()



X[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']]=X[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']].replace(['Ex','Gd','TA','Fa','Po'], [5, 4, 3, 2, 1])
X['Alley']=df['Alley']
X['LotShape']=X['LotShape'].replace(['Reg','IR1','IR2','IR3'],[4,3,2,1])
X['']
df.info()


X['Utilities']=X['Utilities'].replace(['AllPub','NoSewr','NoSeWa','ELO'],[4,3,2,1])
X['BsmtQual']=X['BsmtQual'].replace(['nan'],[0])
X['GarageFinish']=X['GarageFinish'].replace(['Fin','RFn','Unf','NA'],[3,2,1,0])

X['BsmtExposure']= X['BsmtExposure'].replace(['Gd','Av','Mn','No','NA'],[4,3,2,1,0])
X[['BsmtFinType1','BsmtFinType2']]  =X[['BsmtFinType1','BsmtFinType2']].replace(['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA'],[6,5,4,3,2,1,0])         
        
X['CentralAir']=X['CentralAir'].replace(['Y','N'],[1,0])
X['PavedDrive']=X['PavedDrive'].replace(['Y','P','N'],[2,1,0])
X['PoolQC']=df['PoolQC']
X['PoolQC']=X['PoolQC'].replace(['Ex','Gd','TA','Fa','NA'],[4,3,2,1,0])


X.info()
X.columns.to_series().groupby(df.dtypes).groups
X[['Utilities','GarageFinish','BsmtExposure','BsmtFinType1','BsmtFinType2','CentralAir','PoolQC','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','LotShape','PavedDrive']]=X[['Utilities','GarageFinish','BsmtExposure','BsmtFinType1','BsmtFinType2','CentralAir','PoolQC','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','LotShape','PavedDrive']].replace(np.nan,0)

X[['Utilities','GarageFinish','BsmtExposure','BsmtFinType1','BsmtFinType2','CentralAir','PoolQC','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','LotShape','PavedDrive']]=X[['Utilities','GarageFinish','BsmtExposure','BsmtFinType1','BsmtFinType2','CentralAir','PoolQC','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','LotShape','PavedDrive']].astype(int)

b=X.isna().sum()

X['Electrical'].mode()
X['Electrical']=df['Electrical']
X['Electrical']=X['Electrical'].replace(np.nan,'SBrkr')

X['MiscFeature']=X['MiscFeature'].replace(np.nan,'No additional feature')
X['Fence']=X['Fence'].replace(np.nan,'No Fencing')
X['Alley']=X['Alley'].replace(np.nan,'No Alley')
X['GarageType']=X['GarageType'].replace(np.nan,'No Garage')


X['LotFrontage']=X['LotFrontage'].replace(np.nan,X['LotFrontage'].mean())



df3=pd.DataFrame(X[['GarageYrBlt','YearBuilt']])

X['GarageYrBlt']=X['GarageYrBlt'].replace(np.nan,X['YearBuilt'])

df4=pd.DataFrame(X[['MasVnrArea','MasVnrType']])

X['MasVnrType']=X['MasVnrType'].replace(np.nan,'No Masonry')

X['MasVnrArea']=X['MasVnrArea'].replace(np.nan,0)

X=X.drop(columns=['MoSold'])

#analyzing MSZoning patterns with SalePrice to see if any kind of assignment of points can be done.

sbn.catplot(x="SalePrice",y="MSZoning",kind='box',data=df)

sbn.catplot(x="SalePrice",y="Functional",data=df)

sbn.catplot(x="SalePrice",y="Alley",data=df)

sbn.catplot(x="SalePrice",y="HouseStyle",data=df)

sbn.catplot(x="SalePrice",y="Street",kind='box',data=df)

unique1=X.nunique()

X['KitchenAbvGr'].value_counts()

X['total_baths']=X['BsmtFullBath']+X['BsmtHalfBath']+X['FullBath']+X['HalfBath']
X['Porch_area_total']=X['OpenPorchSF']+X['EnclosedPorch']+X['3SsnPorch']+X['ScreenPorch']
X['Garage_overall']
X['Kitchen_Score']
X['Total_area']=X['LotArea']+X['MasVnrArea']+X['BsmtFinSF1']+X['BsmtFinSF2']+X['BsmtUnfSF']+X['TotalBsmtSF']+X['1stFlrSF']+X['2ndFlrSF']+X['LowQualFinSF']+X['GrLivArea']+X['GarageArea']+X['WoodDeckSF']+X['OpenPorchSF']+X['EnclosedPorch']+X['3SsnPorch']+X['ScreenPorch']+X['PoolArea']
X['basement_score']=

X['ext_score']=


X['overall_score']=X['LotShape']+X['Utilities']+X['OverallQual']+X['OverallCond']+X['ExterQual']+X['ExterCond']+X['BsmtQual']+X['BsmtCond']+X['BsmtExposure']+X['BsmtFinType1']+X['BsmtFinType2']+X['HeatingQC']+X['CentralAir']+X['KitchenQual']+X['FireplaceQu']+X['GarageFinish']+X['GarageQual']+X['GarageCond']+X['PavedDrive']+X['PoolQC']

X['HeatingQC'].value_counts()

X1=pd.DataFrame()
X1=X1.drop(columns=['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'])


X1=pd.get_dummies(X1,drop_first=True)
X_train=X1.iloc[0:1460]
X_test=X1.iloc[1460:2919]
from xgboost import XGBRegressor
regressor = XGBRegressor(learning_rate =0.01,n_estimators=1000, max_depth=9,min_child_weight=6,
                         gamma=0,reg_alpha=0.005,subsample=0.8,colsample_bytree=0.8,nthread=4,scale_pos_weight=1,seed=27)
regressor.fit(X_train,y)



y_pred=regressor.predict(X_test)

df5=pd.read_csv('test.csv')
df6=pd.DataFrame()
df6['Id']=df5['Id']

df6['SalePrice']=y_pred

df6.to_csv('predictions_2_adv.csv')

df












from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=1)
model.fit(X[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
        'MoSold', 'YrSold','LotFrontage', 'MasVnrArea', 'GarageYrBlt']],y)
model1=ExtraTreesClassifier(n_estimators=1)
model1.fit(X[['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
        'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']],y)
print(model.feature_importances_) 
print(model1.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
        'MoSold', 'YrSold','LotFrontage', 'MasVnrArea', 'GarageYrBlt']].columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
feat_importances=feat_importances.sort_values(axis=0,ascending = False)


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(X[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
        'MoSold', 'YrSold','LotFrontage', 'MasVnrArea', 'GarageYrBlt']],y)
X_num_scores = pd.DataFrame(fit.scores_)
X_num_columns = pd.DataFrame(X[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
        'MoSold', 'YrSold','LotFrontage', 'MasVnrArea', 'GarageYrBlt']].columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([X_num_columns,X_num_scores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
Numerical_Feature_scores = featureScores.nlargest(35,'Score')  #print 20 best features










x=X[['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
        'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']]

x=pd.get_dummies(x,drop_first = True)

bf1 = SelectKBest(score_func = chi2 , k=150)


fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
categorical_feature_imp = featureScores.nlargest(223,'Score')

xx = X
xx = pd.get_dummies(X,drop_first = True)

bf2 = SelectKBest(score_func = chi2 , k=200)
fit = bf2.fit(xx,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(xx.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
xx_score = featureScores.nlargest(258,'Score')


xx_score['Specs'].groupby('Pool')

print()











#Or creating a separate list for each to get the exact information(it kind of gives the function ing of the library)
a=df['MSZoning'].unique().tolist()
ba=df['Street'].unique().tolist()
ca=df['Alley'].unique().tolist()
da=df['LotShape'].unique().tolist()
ea=df['LandContour'].unique().tolist()
fa=df['Utilities'].unique().tolist()
ga=df['MSZoning'].unique().tolist()
ha=df['LotConfig'].unique().tolist()
ia=df['LandSlope'].unique().tolist()
ja=df['Neighborhood'].unique().tolist()
ka=df['Condition1'].unique().tolist()
la=df['Condition2'].unique().tolist()
ma=df['MSZoning'].unique().tolist()
na=df['BldgType'].unique().tolist()
oa=df['HouseStyle'].unique().tolist()
pa=df['RoofStyle'].unique().tolist()
qa=df['RoofMatl'].unique().tolist()
ra=df['Exterior1st'].unique().tolist()
sa=df['Exterior2nd'].unique().tolist()
ta=df['MasVnrType'].unique().tolist()
ua=df['ExterQual'].unique().tolist()
va=df['ExterCond'].unique().tolist()
wa=df['Foundation'].unique().tolist()
xa=df['BsmtCond'].unique().tolist()
ya=df['BsmtExposure'].unique().tolist()
za=df['BsmtFinType1'].unique().tolist()

b=df['BsmtFinType2'].unique().tolist()
ab=df['Heating'].unique().tolist()
bb=df['HeatingQC'].unique().tolist()
cb=df['CentralAir'].unique().tolist()
db=df['Electrical'].unique().tolist()
eb=df['KitchenQual'].unique().tolist()
fb=df['Functional'].unique().tolist()
gb=df['FireplaceQu'].unique().tolist()
hb=df['GarageType'].unique().tolist()
ib=df['GarageFinish'].unique().tolist()
jb=df['GarageQual'].unique().tolist()
kb=df['GarageCond'].unique().tolist()
lb=df['PavedDrive'].unique().tolist()
mb=df['PoolQC'].unique().tolist()
nb=df['Fence'].unique().tolist()
ob=df['MiscFeature'].unique().tolist()
pb=df['SaleType'].unique().tolist()
qb=df['SaleCondition'].unique().tolist()

corr=df.corr()
sbn.heatmap(corr,annot=True)

X=pd.DataFrame(df)
X=X.drop(columns='SalePrice')

X['No.of_Years_of_construction']=2019-X['YearBuilt']
X['No.of_Years_of_remodelling']=2019-X['YearRemodAdd']
X['Years_sold']=2019-X['YrSold']
X['MoSold']=2019-X['MoSold']
X.count()

#X=X.fillna(X.mean())

X['Functional'].describe()
X['Functional'].mode()
X['Functional'] = X['Functional'].fillna(X['Functional'].mode())

X['Street'].mode()
X['Street'].count()

pd.crosstab(index=df['SalePrice'],columns=X['Street']).plot.hist()


df.groupby('MSSubClass')['MSZoning'].count()
df.groupby('MSZoning')['MSSubClass'].mean()
df.groupby('MSZoning').count()

pd.crosstab(index=df['MSZoning'],columns=df['MSSubClass']).plot.bar()
X['']













from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X=sc.fit_transform(X)

Y=df['SalePrice']

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500, random_state=0)
regressor.fit(X,Y)


df2=pd.read_csv('test.csv')












































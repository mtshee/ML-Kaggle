#Install on personal machine up to date statsmodels:
#pip install -U statsmodels
#Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

#Handle tabular data & matrices
import numpy as np
import pandas as pd

#Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Helpers for Modelling
from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Configure Visualizations
#%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams[ 'figure.figsize'] = 8 , 6

#Helper functions for making good looking plots
def plot_histograms(df, variables, n_rows, n_cols):
    fig = plt.figure(figsize = (16, 12))
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        df[var_name].hist(bins = 10, ax = ax)
        ax.set_title('Skew: ' + str(round ( float( df[var_name].skew() ),) ) )
        ax.set_xticklabels( [], visible = False)
        ax.set_yticklabels( [], visible = False)
        fig.tight_layout()
        plt.show()

def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect = 4, row = row, col = col)
    facet.map(sns.kdeplot, var, shade = True)
    facet.set(xlim =(0, df[var].max() ) )
    facet.add_legend()

def plot_categories(df, car, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row = row, col = col)
    facet.map(sns.barplot, plot_categories, target)
    facet.add_legend()

def plot_correlation_map(df):
    corr = titanic.corr()
    _ , ax = plt.subplots(figsize = (12, 10))
    cmap = sns.diverging_palette(220,10,as_cmap = True)
    _ = sns.heatmap(corr, cmap = cmap, square = True, cbar_kwas = {'shrink': .0}, ax = ax, annot = True, annot_kws = {'fontsize': 12})

def describe_more(df):
    var = []; l = []; t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        l.append(df[x].dtypes)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
    levels.sort_values(by = 'Levels', implace = True)
    return levels

def plot_variable_importance(X, y):
    tree = DecisionTreeClassifier(random_state = 99)
    tree.fit(X, y)
    plot_model_var_imp(tree, X, y)

def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame(model.feature_importances_columns ['Importance'], index = X.columns)
    imp = imp.sort_values(['Importance'], ascending = True)
    imp[:10].plot(kind = 'barh')
    print(model.score(X, y))

def determine_Rich(money):
    if(money['Title'] == "Royalty"):
        return 1
    elif(money['Pclass'] == 1):
        return 1
    elif(money['Fare'] > money['Fare'].quantile(0.75)):
        return 1
    else:
        return 0

def determine_Middle(money):
    if(money['Pclass'] == 2):
        return 1
    elif(money['Fare'].quantile(0.25)<= money['Fare'] <= money['Fare'].quantile(0.75)):
        return 1
    else:
        return 0

def determine_Poor(money):
    if(money['Pclass'] == 3):
        return 1
    elif(money['Fare'] < money['Fare'].quantile(0.25)):
        return 1
    else:
        return 0

def main():
    train = pd.read_csv("../input/train.csv") #change filepath later
    test = pd.read_csv('../input/train.csv') #change filepath later

    full = train.append(testc, ignoer_index = True)
    titanic = full[:891]

    del train, test
    print('Datasets:', 'full:', full_shape, 'titanic:', titanic.shape)

    #Peak at data to see what it looks like
    titanic.head()
    titanic.describe()

    #Plot correlation heat map
    plot_correlation_map(titanic)

    #Plot distribution of Age of passengers
    plot_distribution(titanic, var = 'Age', target = 'Survived', row = 'Sex')

    #Plot distribution of Fare of passengers
    plot_distribution(titanic, var = 'Fare', target = 'Survived', row = 'Pclass')

    #Plot survival rate by embarked
    plot_categories(titanic, cat = 'Embarked', target = 'Survived')

    #Plot survival rate by Sex
    plot_categories(titanic, cat = 'Sex', target = 'Survived')

    #Plot survival rate by Pclass
    plot_categories(titanic, cat = 'Pclass', target = 'Survived')

    #Plot surivival rate by SibSp
    plot_categories(titanic, cat = 'SibSp', target = 'Survived')

    #Plot survival rate by Parch
    plot_categories(titanic, cat = 'Parch', target = 'Survived')

    #Make sex into binary values 0 & 1 (needs to be numerical data)
    sex = pd.Series(np.where(full.Sex == 'male', 1, 0), name = 'Sex')

    #Create new variable for every unique embarked variable
    embarked = pd,get_dummies(full.Embarked, prefix = 'Embarked')
    embarked.head()

    #Create new variable for every unique value of Passenger Class
    pclass = pd.get_dummies(full.Pclass, prefix = 'Pclass')
    pclass.head()

    #Replace 2 missing embarkation values with the port closest to fare value
    
    imputed.head()

    #Extracting title
    title = pd.DataFrame()
    title['Title'] = full['Name'].map( lambda name: name.split(',')[1].split('.')[0].strip())
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Royalty",
        "Lady": "Royalty"
        }
    title['Title'] = title.Title.map(Title_Dictionary)
    title = pd.get_dummies(title.Title)
    #title pd.concat([title, titles_dummies], axis = 1)
    title.head()

    #Replace 1 missing fare value with the median
    full['Fare'] = full.Fare.fillna(full.Fare.median())

    #Fill missing values of Age
    #Option 1: fill with the average of Age
    #Age['Age'] = full.Age.fillna(full.Age.mean())

    #Option 2: use regression analysis to find likely value of age for missing values
    #will need to get rid of negative ages and other stupid values

    #Option 3: fill missing ages with medians that are seperated by group
    stuff['Title'] = title.Title
    stuff['Sex'] = sex
    stuff['Pclass'] = pclass
    stuff['Age'] = full.Age
    stuff['Age'] = stuff.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))

    Age['Age'] = stuff.Age

    del stuff

    #Fill in missing cabin values
    #Use regression from Pclass, ticket, embarkation port, etc...
    cabin = pd.DataFrame()
    

    #Create family size variable
    family = pd.DataFrame()

    family['FamilySize'] = full['Parch'] + full['Sibsip'] + 1

    #Single, small or large family
    family['Family_Single'] = family['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    family['Family_Small'] = family['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    family['Family_Large'] = family['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    family.head()

    #Create a wealth variable
    wealth = pd.DataFrame()
    money = pd.DataFrame()
    money['Pclass'] = full['Pclass']
    money['Title'] = title['Title']
    money['Fare'] = full['Fare']
    cabin['Cabin'] = full['Cabin']

    wealth['Social_Class'] 

    #Create Functions to define if Poor, Middle Class or Rich
    wealth['Poor'] = wealth['Social_Class'].map(determine_Poor(money))
    wealth['Middle_Class'] = wealth['Social Class'].map(determine_Middle(money))
    wealth['Rich'] = wealth['Social Class'].map(determine_Rich(money))

    full_X = pd.concat([Age, embarked, cabin, sex, wealth, family], axis = 1)
    full_X.head()

    #Create all datasets neccessary to test models
    train_valid_X = full_X[0:891]
    train_valid_Y = titanic.Survived
    test_X = full_X[891:]
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_valid_X, train_valid_Y, train_size = 0.7)
    print (full_X.shape, train_X.shape, valid_X.shaoe, train_Y.shape, valid_Y.shape, test_X.shape)

    plot_variable_importance(train_X, train_Y)

    #Run several different models
    model1 = RandomForestClassifier(n_estimators = 100)
    model2 = SVC()
    model3 = GradientBoostingClassifier(n_neighbors = 3)
    model4 = GaussianNB()
    model5 = LogisticRegression()

    model1.fit(train_X, train_Y)
    model2.fit(train_X, train_Y)
    model3.fit(train_X, train_Y)
    model4.fit(train_X, train_Y)
    model5.fit(train_X, train_Y)

    train_score1 = model1.score(train_X, train_Y)
    train_score2 = model2.score(train_X, train_Y)
    train_score3 = model3.score(train_X, train_Y)
    train_score4 = model4.score(train_X, train_Y)
    train_score5 = model5.score(train_X, train_Y)
    
    valid_score1 = model1.score(valid_X, valid_Y)
    valid_score2 = model2.score(valid_X, valid_Y)
    valid_score3 = model3.score(valid_X, valid_Y)
    valid_score4 = model4.score(valid_X, valid_Y)
    valid_score5 = model5.score(valid_X, valid_Y)

    #Print out score comparisons
    print("Train Data Score: Validation Data Score:")
    print(train_score1, valid_score1)
    print(train_score2, valid_score2)
    print(train_score3, valid_score3)
    print(train_score4, valid_score4)
    print(train_score5, valid_score5)

    #Hopefully find the Optimal Features for the model
    plot_model_var_imp(model1, train_X, train_Y)
    rfecv = RFECV(estimator = model1, step = 1, cv = StratifiedKFold(train_Y, 2), scoring = 'accuracy')
    rfecv.fit(train_X, train_Y)
    print(rfecv.score(train_X, train_Y), rfecv.score(valid_X, Valid_Y))
    print("Optimal number of features: %d" % refecv.n_features_)

    #Plot number of features vs. cross Validcation Scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classification")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores)
    plt.show()

if __name__ == "__main__":
    main()






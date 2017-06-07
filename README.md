# TitanicML-Kaggle
My Work on the Intro to Machine Learning Competition on the Titantic on Kaggle

Things to edit to make more accurate:
1. Can I just fill in the missing age values instead of replacing the entire variable
   Or is there an even more accurate model for the age values?
   Regression model to get missing age values? (Linear or otherwise?)
2. Seperate male and female adults to see if that changes survivability
   It should change it quite a bit
   Children, Mother, Female Adult, Male Adult groups
3. Is there a more applicable/accurate algorithm for the model?
   Are there different variables that can be added to the model?
   Are there some variables that should be taken away to improve accuracy?
4. How prominent is the family size variable? Could it be more prominent?
5. Consider ticket numbers as a possible variable
   Those with same ticket numbers, traveling together?
6. Create a Variable that predicts survival based upon income
   Combine port, cabin, class, & fare to determine wealth
   Will definitely need to do some manipulation but can possibly help alot
7. Group family size by ticket instead of Parch & SibSp
8. Combine & stack multiple prediction models to yield a more accurate result
9. How to acocunt for dependence among the variables? Are there redundancies?
   Sex and Age are very related to title, not very independent

Most Important Variables:
Sex
Age
Title (maybe redundant?)
Wealth (see point 6)



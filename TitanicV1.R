#Import libraries for data visualization, manipulation & classification
library('ggplot2')
library('ggthemes')
library('ggscales')
library('dplyr')
library('mice')
library('randomForest')

#Things to edit to make more accurate:
#1. Can I just fill in the missing age values instead of replacing the entire variable
#   Or is there an even more accurate model for the age values?
#2. Seperate male and female adults to see if that changes survivability
#   It should change it quite a bit
#   Children, Mother, Female Adult, Male Adult groups
#3. Is there a more applicable/accurate algorithm for the model?
#   Are there different variables that can be added to the model?
#   Are there some variables that should be taken away to improve accuracy?
#4. How prominent is the family size variable? Could it be more prominent

#Part 1: Data Importation & Reorganization

#import data
train <- read.csv('../input/train.csv', stringsAsFactors = F)
test <- read.csv('../input/test.csv', stringAsFactors = F)

full <- bind_rows(train, test) #bind the data together (like a tuple)

#Check data
str(full)

#We should have the following variables:
#Passenger class, Passenger Name, Passenger Sex, Passenger Age, #of Siblings/spouses aboard, #of parents/children aboard
#Ticket Number, Fare, Cabin, Port of Embarkation

#Parse passenger title from the passenger name
full$title <- gsub('(.*,)|(\\..*', '', full$Name)

#Show the different titles by sex
table(full$Sex, full$Title)

#Titles that have low counts are combined to a single, "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 
    'Major', 'Rev', 'Sir', 'Jonkheer')

#Rename unusual title names to their similar counterparts
full$Title[full$Title == 'Mlle'] <- 'Miss'
full$Title[full$Title == 'Ms'] <- 'Miss'
full$Title[full$Title == 'Mme'] <- 'Mrs'
full$Title[full$Title %in% rare_title] <- 'rare_title'

#Grab Last (Family) name from passenger name
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])

#Part 2: Determine how family size contributes to survivability

#Create a family size variable (include the passenger themeselves)
full$Fsize <- full$SibSp + full$Parch + 1

#Create a family variable
full$Family <- paste(full$Surname, full$Fsize, sep = '_')

#Plot family size vs. survival histogram (use ggplot2)
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
geom_bar(stat = 'count', position = 'dodge') +
scale_x_continous(breaks = c(1:11)) +
labs(x = 'Family Size') +
theme_few()

#Data should show that there is a survival penalty for those that are single
#Should also show there is a survival penality for those with family size > 4

#Break family size into 3 levels

#Discretized (discrete) family size variable
full$FsizeD[full$Fsize == 1] <- 'Single'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

#Show how family size relates to survival with a mosaic plot
mosaicplot(table(full$FsizeD, full$Survived), main = 'Family Size by Survival', shade = TRUE)

#Define a passenger Deck variable (better Deck, better survival rate?)
full$Deck <- factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

#Part 3: Fill in missing data values based off of statistics & predictive models

#Get rid of missing passenger IDs
embark_fare <- full %>% filter(PassengerId != 62 & PassengerId != 830)

#Plot embarkment, passenger class and median fare
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
geom_boxplot() +
geom_hline(aes(yintercept = 80),
colour = 'red', linetype = 'dashed', lwd = 2) +
scale_y_continuous(labels = dollar_format()) +
theme_few()

#First class from Charbourg very close to $80 median, most likely left from 'C'
full$Embarked[c(62, 830)] <- 'C'

#Missing fare value in row 1044
#Visualize fares sharing class 3 and embarkment from Southhampton
ggplot(full[full$Pclass == '3' & full$Embarked == 'S',],
aes(x = Fare)) + geom_density(fill = '#99d6ff', alpha = 0.4) +
geom_vline(aes(xintercept = median(Fare, na.rm = T)),
colour = 'red', linetype = 'dashed', lwd = 1) +
scale_x_continuous(labels = dollar_format()) + theme_few()

#Replace missing fare value with the median for class/embarkment
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S',]$Fare, na.rm = TRUE)

#Predict the missing age values based upon mice package

#Make variables factors into factors
factor_vars <- c('PassengerId', 'Pclass', 'Sex', 'Embarked',
    'Title', 'Surname', 'Family', 'FsizeD')
full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

#Set a random seed
set.seed(129)

#Perform mice imputation, excluding variables known to be unhelpful
mice_mod <- mice(full[, !names(full) %in%
c('PassengerId', 'Name', 'Ticket', 'Cabin', 'Family', 'Surname', 'Survived')], method = 'rf')

#Save complete output
mice_output <- complete(mice_mod)

#Plot age distributions
par(mfrow = c(1, 2))
hist(full$Age, freq = F, main = 'Age: Original Data', col = 'darkgreen', ylim = c(0, 0.04))
hist(mice_output$Age, freq = F, main = 'Age: MICE Output', col = 'lightgreen', ylim = c(0, 0.04))

#Replace Age variable from the mice model
full$Age <- mice_output$Age

#Part 4: prediction of survival (rely on randomForest classification algorithm)

#Split data back into original test and training sets
train <- full[1:891,]
test <- full[892:1309,]

#Set a random seed
set.seed(754)

#Build model (not all possible variables used at this time (something to change))
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked
    + Title + FsizeD + Child + Mother, data = train)

#Show model error
plot(rf_model, ylim = c(0, 0.36))
legend('topright', colnames(rf_model$err.rate), col = 1:3, fill = 1:3)

#Get variable importance
importance <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance),
    Importance = round(importance[, 'MeanDecreaseGini'], 2))

#Create a rank variable based on importance
rankImportance <- varImportance %>%
mutate(Rank = paste0('#', dense_rank(desc(importance))))

#Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, importance),
y = importance, fill = importance)) +
geom_bar(stat = 'identity') +
geom_text(aes(x = Variables, y = 0.5), label = Rank,
hjust = 0, vjust = 0.55, size = 4, colour = 'red') +
labs(x = 'Variables') +
coord_flip() +
theme_few()

#Predict using test set
prediction <- predict(rf_model, test)

#Save solution to dataframe with Id & Survived columns
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

#Write solution to a file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)
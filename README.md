![img](https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Bondora_logo.svg/1200px-Bondora_logo.svg.png?20210113165119)
# Credit Risk Analysis For Bondora(P2P Lending Business)
In this project, the objective is to leverage predictive modeling techniques to enhance the decision-making process for P2P lending. The project will have two main phases: loan default prediction and investment strategy optimization. The goal is to provide insights and tools that enable more informed lending decisions, improved ROI prediction, and efficient fund allocation for both lenders and borrowers.

## Table of Contents
- [Abstract](#abstract)
- [Background Of Understanding The Problem](#background-of-understanding-the-problem)
- [Project lifecycle](#project-lifecycle)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [EDA](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Building Pipeline](#building-pipeline)
- [App Creation](#app-creation)
- [Deployment](#deployment)
- [Challenges and Limitations](#challenges-and-limitations)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [Further Reading and Resources](#further-reading-and-resources)
- [Contact](#contact)

## Abstract
In this project we will be doing credit risk modelling of peer to peer lending Bondora systems.Data for the study has been taken from a leading European P2P lending platform (Bondora).The retrieved data is a pool of both defaulted and non-defaulted loans from the time period between 1st March 2009 and 27th January 2020. The data comprises of demographic and financial information of borrowers, and loan transactions.In P2P lending, loans are typically uncollateralized and lenders seek higher returns as a compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision, and realize the return that compensates for the risk.

## Background of Understanding the Problem
Peer-to-peer lending has attracted considerable attention in recent years, largely because it offers a novel way of connecting borrowers and lenders. But as with other innovative approaches to doing business, there is more to it than that. Some might wonder, for example, what makes peer-to-peer lending so different–or, perhaps, so much better–than working with a bank, or why has it become popular in many parts of the world.
Certainly, the industry has witnessed strong growth in recent years. According to Business Insider, transaction volumes in the U.S. and Europe, the world’s leading P2P markets, have expanded at double and, in some cases, triple-digit percentage rates, bolstered by widespread acceptance of doing business online and a supportive regulatory environment.
For investors, "peer-2-peer lending," or "P2P," offers an attractive way to diversify portfolios and enhance long-term performance. When they invest through a peer-to-peer platform, they can profit from an asset class that has proven itself in both good times and bad. Equally important, they can avoid the risks associated with putting all their eggs in one basket, especially at a time when many experts believe that traditional favorites such as stocks and bonds are riskier than ever.
Default risk has long been a significant risk factor to test borrowers’ behaviour in Peer-to-Peer (P2P) lending. In P2P lending, loans are typically uncollateralized and lenders seek higher returns as compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision and realize the return that compensates for the risk.
As in the financial research domain, there are very few datasets available that can be utilized for building and analyzing credit risk models. This dataset will help the research community in building and performing research in the credit risk domain.

## Project lifecycle
![img_29](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/68d90c1e-56f6-4828-9220-6d5d64a2c5df)


## Data Collection
- Data for the study has been retrieved from a publicly available data set of a leading European P2P lending platform (Bondora).
- In order to maintain a clean dataset, we filtered out irrelevant or spammy posts and focused on gathering data that pertained to the stock market. This ensured that our dataset consisted of high-quality data points that could provide meaningful insights for our analysis.

## Data Preprocessing
- The dataset contains 112 Columns and 134529 Rows Range Index
- Removing the columns having missing value for then 40% missing values . after remove null columns now we have 77 columns and 134529 Rows for further use.
![img_3](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/ba0cefff-58b3-4617-afcd-c2c57a532d5f)

- Apart from missing value features there are some features which will have no role in default prediction like 'ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC', 'DateOfBirth', 'BiddingStartedOn', 'UserName', 'NextPaymentNr', 'NrOfScheduledPayments', 'IncomeFromPrincipalEmployer', 'IncomeFromPension', 'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare', 'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther', 'LoanApplicationStartedDate', 'ApplicationSignedHour', 'ApplicationSignedWeekday','ActiveScheduleFirstPaymentReached', 'PlannedInterestTillDate', 'LastPaymentOn', 'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn', 'ProbabilityOfDefault', 'PrincipalOverdueBySchedule', 'StageActiveSince', 'ModelVersion','WorseLateCategory', and all the date columns except the 'DefaultDate'.
- We will not conside the 'Current' in the Status column as their EMI is still going on.

    ![img](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/c7a193c1-211e-4c77-a170-26a494cfdd5b)

- We will replace all the default date in DefaultDate column as 'Default' and all the Nan values with 'NonDefault', as the date represents the day when the loan got defaulted.
![img_1](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/a65a356f-4762-4acf-980e-a5e02d7c1463)

- We will also filter out ages above 18years.

    ![img_2](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/3935fa62-8cad-4591-ae64-721188f836ba)

- Now we have 31 columns and 77341 Rows for further use.
- Null value treatment was done by imputing continuous variables with median and categorical variables with mode.
![img_4](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/4e427f5e-f111-4a05-b83f-77bef25f6161)

- Outlier treatement was performed by capping the extreme values with upper and lower bound of the data, for this we utilized the IQR mehtod.
![img_5](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/db95c7a0-c09b-40da-9ba8-99a0b7232e05)

- Creating target features(EMI, ELA & PROI)
![img_19](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/1f4bdb80-a6ef-4bb3-b422-50c35e423d9e)

- Now, we have a clean optimized dataset, and we're ready for EDA.

## Exploratory Data Analysis
Upon visualizing the dataset, intriguing patterns emerge, as the subsequent images vividly illustrate...
- Percentage of default and non default borrowers

    ![img_6](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/ca77c5c2-a3b3-457a-8b47-1ae0e8362303)

- Gender distribution in defaulting borrowers

    ![img_7](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/8eee6bb7-16c3-43e9-86bb-85d848865c12)

- Percentage of new borrowers in defaulting borrowers

    ![img_8](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/4d2c513e-582f-4867-9e40-b96e117a5bee)

- Distribution of education qualification of defaulting borrowers

    ![img_9](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/68b95181-9898-40c9-ba34-cf63df8791a5)

- Distribution of employment status in defaulting borrowers

    ![img_10](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/ef05de25-c32b-406a-a773-7562c8eb1a5b)

- Distribution of employment duration with current employer in defaulting borrowers

    ![img_11](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/a3c513f8-5dda-4edd-aa21-20a6a4cb4672)

- Distribution of age in defaulting borrowers

    ![img_12](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/435d4683-9e77-4e09-8c11-ae3bd19305cb)

- Distribution of monthly payment in defaulting borrowers

    ![img_13](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/9c171bda-8504-4190-ac0a-44346398fa77)


## Feature Engineering
- Feature engineering played a vital role in our project, as it allowed us to extract meaningful insights from the raw text data and transform it into a format that could be easily understood by our machine learning models. By carefully selecting and engineering relevant features, we were able to improve the performance of our models and enhance the overall effectiveness of our predictions.
  - Here we followed the following steps:- 
    - Feature Encoding(Used Label Encoder to encode all the categorical features)
    
      ![img_16](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/9d4c771a-b28c-44ea-aa90-7675653222d1)
 
    - Splitting into train and test sets
    
      ![img_20](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/cf8f7b42-d2a1-4c3e-b2ec-64c617c131f5)

    - Addressing the class imbalace problem(SMOTE Analysis)
    
      ![img_14](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/0af96c35-44f2-49be-8a7d-85d637a55888)

    - Standard Scaling
    
      ![img_15](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/790dd87b-0f8c-49e3-8c33-df39d2f5ced4)

    - PCA(Here we proceed with all the feature in the feature space)
    
      ![img_17](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/7c41d345-f0b9-4c4f-8121-4172adb28c6d)


## Model Building
### Classification Model
We experimented with two machine learning models to predict the LoanStatus('Default' or 'NonDefault')
- We used classification_report(precision | recall | f1-score ), confusion_matrix, accuracy_score, and roc_auc_score metrics from sklearn.metrics to asses each model.
- The models used for classification are...
  - Logistic Regression
  - Random Forest
- Each model was trained using the engineered features, and their performance was evaluated on a test dataset. This iterative process enabled us to identify the strengths and weaknesses of each model and select the most suitable one for our project.
- Out of the two models Random Forest was giving better results:

    ![img_18](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/6f0ad8cd-0b95-47f0-8f49-055418d09531)

  
### Regression Model
We experimented with two machine learning models to predict EMI, ELA, PROI in a mulitple output regression process.
- We used r2 score metrics from sklearn.metrics to asses each model.
- The models used for regression are...
  - Linear Regression
  - Lasso Regression
- Each model was trained using the engineered features, and their performance was evaluated on a test dataset. This iterative process enabled us to identify the strengths and weaknesses of each model and select the most suitable one for our project.
 - Out of the two models Linear Regression was giving better results:

    ![img_21](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/81a7e513-1c75-4d10-8997-eca006292b73)


## Hyperparameter Tuning
To further improve the performance of our chosen model, we performed hyperparameter tuning. This process involved adjusting various parameters within the model to optimize its performance on the training data. By fine-tuning the model's hyperparameters, we were able to achieve better results and enhance the overall accuracy of our predictions.
![img_22](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/eda524ef-e2b5-45d9-9bca-1a6522afa44e)


## Building Pipeline
![img_28](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/de3209c0-4b3b-4476-bb26-d55c42ce845a)


Once we had selected and fine-tuned our model, we proceed with building of pipeline.
- Classification Pipeline

    ![img_23](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/4e379526-2db9-47a7-a5ba-c08137b2f313)

- Regression Pipeline

    ![img_24](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/0c68fdde-2f6e-4522-a9e9-72e6066df6be)

- Saving pipelines

    ![img_25](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/525d9e0c-7429-4840-baac-337c957ee545)

- Combining pipelines

    ![img_26](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/951736ad-15cd-4ed4-9b4f-bf57e031a550)


## App Creation
- Using Flask, HTML5, and CSS; we created a web application.
  1. app.py --> for Flask to run the app and preprocess the input data from the web forum and make it match the attributes expected by the pipeline file and furthermore run the pipeline files. 
  2. index.html --> Individual borrower entry using a web forum.
- UI
  - To facilitate user interaction with the model, we designed a simple and user-friendly front-end interface using HTML, CSS, and JavaScript. This interface allow investors to input borrower data and receive predictions based on the model's analysis.

## Deployment
- After combining and saving our pipelines, we provided detailed steps and code for deploying the model. This included creating a Flask application, and deploying the model on our local system. This allowed investors to access the model and make predictions using the data they input and the model running at the background.
- UI
  
    ![img_27](https://github.com/akhilraj7000/Bondora_Credit_Risk_Model/assets/110608974/65c3c45e-0191-4abf-8522-71490d31a63a)


## Challenges and Limitations
Our project faced several challenges and limitations, including:

- The quality and relevance of the data collected.
- The inherent complexity of borrower data.
- The limitations of ML model in capturing nuanced and context-dependent information.
- The potential overfitting of our models.

Despite these challenges, our project demonstrates the potential of using ML on loan applicant data to predict credit risk.

## Future Work
There are several areas of improvement and expansion for this project:
- Exploring additional feature engineering techniques to improve model performance
- Investigating more advanced models, such as XGBoost and neural networks.

## Acknowledgements
We would like to thank the following resources and libraries for their invaluable contribution to this project:
- [Scikit-learn](https://scikit-learn.org/)
- [Target creation report by Nour Ibrahim]()
- [Flask](https://flask.palletsprojects.com/)
- [Multiple Output Regression by Krish Naik](https://www.youtube.com/watch?v=26J3bcqhfLE)
- [Pipeline Creation by CampusX](https://www.youtube.com/watch?v=xOccYkgRV4Q)

## Further Reading and Resources
For those interested in learning more about P2P lending and finance, we recommend the following resources:
- [Peer-to-Peer Lending](https://www.investopedia.com/terms/p/peer-to-peer-lending.asp)
- [Machine Learning for Finance](https://www.packtpub.com/product/machine-learning-for-finance/9781789136364) by Jannes Klaas

## Contact
If you have any questions or suggestions regarding this project, feel free to reach out to us:

- Malavika Lakshmipriya: [malavikalp12@gmail.com](malavikalp12@gmail.com)
- Akhil Raj CV: [akhilraj7000@gmail.com](akhilraj7000@gmail.com)
- Apurva Singh: [k.apurva304@gmail.com](k.apurva304@gmail.com)
- Shivendra Mane Deshmukh: [shivendramanedeshmukh0712@gmail.com](shivendramanedeshmukh0712@gmail.com)
- Sidhant Bhosale: [sidhantbhosale2019@gmail.com](sidhantbhosale2019@gmail.com)
- Shreyash Deolalikar: [deolalikarshreyash143@gmail.com](deolalikarshreyash143@gmail.com)
- Sahil Narkar: [sahilnarkar1551997@gmail.com](sahilnarkar1551997@gmail.com)
- Atul Singh: [atulsingh2317@gmail.com](atulsingh2317@gmail.com)
- Kumari Ritul: [ritulkumari18@gmail.com](ritulkumari18@gmail.com)
- Arya Dhane: [aryadhane27@gmail.com](aryadhane27@gmail.com)
- Ramesh Bathula: [bathularamesh13@gmail.com](bathularamesh13@gmail.com)

We appreciate your interest in our project and look forward to hearing your thoughts and feedback!


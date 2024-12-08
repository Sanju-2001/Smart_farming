import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score, classification_report

crop_data = pd.read_csv('Crop_recommendation.csv')
print(crop_data.shape)

#Encoding target column text labels to numbers
unique_crops = crop_data['label'].unique().tolist()
unique_crops_index = np.arange(0,len(unique_crops),1)
unique_crops_text_to_num = {i:j for i,j in zip(unique_crops, unique_crops_index)}
unique_crops_num_to_text = {j:i for i,j in zip(unique_crops, unique_crops_index)}
crop_data['label'] = crop_data['label'].map(unique_crops_text_to_num)

crop_data = crop_data.sample(frac=1)#shuffling the dataset

x = crop_data.iloc[:,:-1].values#choosing input columns
y = crop_data.iloc[:,-1].values#choosing output columns

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=123,test_size=0.2)


st.title('CROP RECOMMENDATION USING MACHINE LEARNING')
with st.sidebar:
    page_option = option_menu('NAVIGATION',
                              options=['HOME','EDA','MODELS','RECOMMENDATION'],
                              icons=['house-fill', 'search', 'clipboard-data-fill', 'clipboard-check'])

if page_option == 'HOME':
    st.write('Agriculture plays a pivotal role in sustaining global food security and economic prosperity. However, farmers face challenges in selecting the most suitable crops for their specific agricultural conditions, leading to suboptimal yields and resource utilization. In this project, we propose a comprehensive solution leveraging machine learning techniques to assist farmers in making informed crop selection decisions. We explore the efficacy of three powerful algorithms: logistic regression, XGBoost, and AdaBoost, in predicting the optimal crops based on various environmental and soil factors. The dataset comprises historical agricultural data, including soil properties, climate conditions, and crop yields. We preprocess the data, perform feature engineering, and train the models using cross-validation techniques to ensure robustness and generalization. The performance of each algorithm is evaluated based on accuracy, precision, recall, and F1-score metrics.')

    st.image(Image.open('images_farming/0_agriculture.jpg'))

if page_option == 'EDA':
    st.subheader('Exploratory Data Analysis(EDA)')
    st.header('Violin plot')
    st.image(Image.open('images_farming/eda_1.png'))

    st.header('Box plot')
    st.image(Image.open('images_farming/eda_2.png'))

    st.header('Pie Chart')
    st.image(Image.open('images_farming/eda_3.png'))

    st.header('Count Plot')
    st.image(Image.open('images_farming/eda_4.png'))

if page_option == 'MODELS':
    st.header('LOGISTIC REGRESSION')
    st.success('Logistic regression is a statistical method used for binary classification tasks, where the goal is to predict the probability that an instance belongs to a particular class. Despite its name, logistic regression is primarily used for classification rather than regression problems. It models the relationship between one or more independent variables and the probability of the outcome occurring using the logistic function. This function maps any real-valued number into a value between 0 and 1, making it suitable for modeling probabilities. Logistic regression is widely used in various fields, including medicine, economics, and social sciences, due to its simplicity, interpretability, and effectiveness, especially when dealing with linearly separable data.')
    st.image(Image.open('images_farming/1_logistic_regression.png'))
    
    st.header('XGBOOST')
    st.error("XGBoost, short for eXtreme Gradient Boosting, is an advanced implementation of gradient boosting algorithms designed to optimize accuracy and computational speed. It is a supervised learning technique used for regression and classification problems. XGBoost builds a series of decision trees iteratively, where each new tree corrects the errors made by the previous one. It employs a gradient descent algorithm to minimize a loss function, enhancing the model's predictive power with each iteration. XGBoost is renowned for its efficiency, scalability, and performance, often outperforming other machine learning algorithms in various competitions and real-world applications. Its versatility and capability to handle large datasets with high dimensionality make it a popular choice among data scientists and machine learning practitioners.")
    st.image(Image.open('images_farming/2_xgboost.png'))

    st.header('ADABOOST')
    st.warning("AdaBoost, short for Adaptive Boosting, is a popular ensemble learning technique used for classification and regression tasks. It works by combining multiple weak learners, typically decision trees, to create a strong learner. In each iteration, AdaBoost assigns higher weights to misclassified data points, allowing subsequent weak learners to focus more on these difficult instances. This iterative process continues until a predefined number of weak learners are added or until a perfect prediction is achieved. AdaBoost adjusts the contribution of each weak learner based on its performance, emphasizing the importance of difficult-to-classify instances. Despite its simplicity, AdaBoost often yields impressive results and is widely used in various machine learning applications for its effectiveness in handling complex datasets and improving predictive performance.")
    st.image(Image.open('images_farming/3_adaboost.png'))

if page_option == 'RECOMMENDATION':
    N=st.number_input("ENTER Nitrogen VALUE ",min_value=1)
    P=st.number_input('ENTER Phosphorus VALUE ',min_value=1)
    K=st.number_input("ENTER Potassium  VALUE ",min_value=1)
    temperature=st.number_input("ENTER AVERAGE TEMPERATURE",min_value=1)
    humidity=st.number_input("ENTER AVERAGE HUMIDITY",min_value=1)
    ph=st.number_input("ENTER PH OF SOIL",min_value=1)
    rainfall=st.number_input("ENTER AVERAGE RAINFALL",min_value=1)

    submit_button=st.button("SUBMIT")
    st.markdown('---')

    if submit_button:
        user_inputs = [[N, P,K, temperature, humidity, ph, rainfall]]

        st.header('MODEL RECOMMENDATION')
        tab1, tab2, tab3 = st.tabs(['LOGISTIC REGRESSION', 'XGBOOST', 'ADABOOST'])

        with tab1:
            from sklearn.linear_model import LogisticRegression
            logreg = LogisticRegression()
            logreg.fit(x_train, y_train)

            logreg_pred = logreg.predict(x_test)
            logreg_output = logreg.predict(user_inputs)[0]

            st.info(f'The best crop for your soil type is : **{unique_crops_num_to_text[logreg_output]}**')

            st.header('LOGISTIC REGRESSION MODEL PARAMETERS')
            st.success(f"ACCURACY : {round(accuracy_score(y_test, logreg_pred),2)*100}%")

            logreg_classification_report = classification_report(y_test, logreg_pred, output_dict=True)
            st.dataframe(logreg_classification_report)

        with tab2:
            from xgboost import XGBClassifier
            xgb = XGBClassifier()
            xgb.fit(x_train, y_train)

            xgb_pred = xgb.predict(x_test)
            xgb_output = xgb.predict(user_inputs)[0]

            st.warning(f'The best crop for your soil type is : **{unique_crops_num_to_text[xgb_output]}**')
            
            st.header('XGBOOST MODEL PARAMETERS')
            st.success(f"ACCURACY : {round(accuracy_score(y_test, xgb_pred),2)*100}%")

            xgb_classification_report = classification_report(y_test, xgb_pred, output_dict=True)
            st.dataframe(xgb_classification_report)

        with tab3:
            from sklearn.ensemble import AdaBoostClassifier
            adaboost = AdaBoostClassifier()
            adaboost.fit(x_train, y_train)

            adaboost_pred = adaboost.predict(x_test)
            adaboost_output = adaboost.predict(user_inputs)[0]

            st.error(f'The best crop for your soil type is : **{unique_crops_num_to_text[adaboost_output]}**')
            
            st.header('ADABOOST MODEL PARAMETERS')
            st.success(f"ACCURACY : {round(accuracy_score(y_test, adaboost_pred),2)*100}%")

            adaboost_classification_report = classification_report(y_test, adaboost_pred, output_dict=True)
            st.dataframe(adaboost_classification_report)

    else:
        st.error('click submit button')
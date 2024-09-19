# +
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from datetime import datetime
# from IPython.display import display
from streamlit_folium import st_folium
from streamlit_folium import folium_static
from streamlit_navigation_bar import st_navbar
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly.subplots import make_subplots


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
# import scikitplot as skplt

# %pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')
# -

st.set_page_config(layout='wide')

bank = pd.read_csv('BankChurners.csv')

pd.set_option('display.max_columns', None)

# **Streamlit**

# Start of Pandas Profiling process
start_time = dt.datetime.now()
print('Started at: ', start_time)

# +
# plot(bank)
# -

#create tabs
tab1, tab2 = st.tabs(['Churns Overview', 'ML Model'])

# **Age Distribution**

with tab1:
    
    # Center the title using HTML inside Markdown
    st.markdown("<h3 style='text-align: center;'>Distribution of Customer Age</h3>", unsafe_allow_html=True)
    
    fig1 = go.Figure(go.Box(x=bank['Customer_Age'], name='Age Box Plot', boxmean=True))
    fig2 = go.Figure(go.Histogram(x=bank['Customer_Age'], name='Age Histogram'))

    # Create two columns in Streamlit
    col1, col2 = st.columns(2)

    # Display the plots side by side
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        
    st.write('We can see that the distribution of customer ages in our dataset follows a fairly noraml distribution; thus, further use of age feature can be done with the normality assumption.')

# **Platinum vs Blue cards**

with tab1:
    st.markdown("<h3 style='text-align: center;'>Distribution Of Gender And Different Card Statuses</h3>", unsafe_allow_html=True)
    fig = make_subplots(
    rows=2, cols=2,subplot_titles=('','<b>Platinum Card Holders','<b>Blue Card Holders<b>','Residuals'),
    vertical_spacing=0.09,
    specs=[[{"type": "pie","rowspan": 2}       ,{"type": "pie"}] ,
           [None                               ,{"type": "pie"}]            ,                                      
          ]
    )

    fig.add_trace(
        go.Pie(values=bank.Gender.value_counts().values,labels=['<b>Female<b>','<b>Male<b>'],hole=0.3,pull=[0,0.3]),
        row=1, col=1
    )

    fig.add_trace(
        go.Pie(
            labels=['Female Platinum Card Holders','Male Platinum Card Holders'],
            values=bank.query('Card_Category=="Platinum"').Gender.value_counts().values,
            pull=[0,0.05,0.5],
            hole=0.3

        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Pie(
            labels=['Female Blue Card Holders','Male Blue Card Holders'],
            values=bank.query('Card_Category=="Blue"').Gender.value_counts().values,
            pull=[0,0.2,0.5],
            hole=0.3
        ),
        row=2, col=2
    )



    fig.update_layout(
        height=800,
        showlegend=True,
    )

    st.plotly_chart(fig)
    st.write('More samples of females in our dataset are compared to males, but the percentage of difference is not that significant, so we can say that genders are uniformly distributed.')

# **Distribution of Dependent counts(close family size)**

with tab1:
    # Center the title using HTML inside Markdown
    st.markdown("<h3 style='text-align: center;'>Distribution of Dependent counts(close family size)</h3>", unsafe_allow_html=True)
    
    fig1 = go.Figure(go.Box(x=bank['Dependent_count'], name='Dependent count Box Plot', boxmean=True))
    fig2 = go.Figure(go.Histogram(x=bank['Dependent_count'], name='Dependent count Histogram'))

    # Create two columns in Streamlit
    col1, col2 = st.columns(2)

    # Display the plots side by side
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        
    st.write('The distribution of Dependent counts is fairly normally distributed with a slight right skew.\n')

# **Education & Marital levels**

with tab1:
    # Create the first Pie chart for Education levels
    fig1 = go.Figure(go.Pie(labels=bank['Education_Level'].value_counts().index, 
                            values=bank['Education_Level'].value_counts().values,
                            hole=0.33))
    fig1.update_layout(title={'text': 'Proportion Of Education Levels', 'x':0.4, 'xanchor':'center'})

    # Create the second Pie chart for Marital Status
    fig2 = go.Figure(go.Pie(labels=bank['Marital_Status'].value_counts().index, 
                            values=bank['Marital_Status'].value_counts().values,
                            hole=0.33))
    fig2.update_layout(title={'text': 'Proportion Of Different Marriage Statuses', 'x':0.43, 'xanchor':'center'})
    
    # Create two columns in Streamlit
    col1, col2 = st.columns(2)

    # Display the plots side by side
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        
    st.write('If most of the customers with unknown education status lack any education, we can state that more than 70% of the customers have a formal education level. About 35% have a higher level of education.')
    st.write('Almost half of the bank customers are married, and interestingly enough, almost the entire other half are single customers. only about 7% of the customers are divorced, which is surprising considering the worldwide divorce rate statistics! (let me know where the bank is located and sign me up!)\n')

# **Income & Card Category**

with tab1:
    # Create the first Pie chart for different income levels
    fig1 = go.Figure(go.Pie(labels=bank['Income_Category'].value_counts().index, 
                            values=bank['Income_Category'].value_counts().values,
                            hole=0.33))
    fig1.update_layout(title={'text': 'Proportion Of Different Income Levels', 'x':0.4, 'xanchor':'center'})

    # Create the second Pie chart for differen
    fig2 = go.Figure(go.Pie(labels=bank['Card_Category'].value_counts().index, 
                            values=bank['Card_Category'].value_counts().values,
                            hole=0.33))
    fig2.update_layout(title={'text': 'Proportion Of Different Card Categories', 'x':0.43, 'xanchor':'center'})
    
    # Create two columns in Streamlit
    col1, col2 = st.columns(2)

    # Display the plots side by side
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        
    st.write('If we include the Unknown to the less than $40k then almost 46.2% earn less than 40k')
    st.write("I believe the that the 'Blue' card is the basic one because 93.2% of clients use the card, that's a huge amount but totally understunable\n")

# **Distribution of months**

with tab1:
    # Center the title using HTML inside Markdown
    st.markdown("<h3 style='text-align: center;'>\nDistribution of months the customer is part of the bank</h3>", unsafe_allow_html=True)
    
    fig1 = go.Figure(go.Box(x=bank['Months_on_book'], name='Months on book Box Plot', boxmean=True))
    fig2 = go.Figure(go.Histogram(x=bank['Months_on_book'], name='Months on book Histogram'))

    # Create two columns in Streamlit
    col1, col2 = st.columns(2)

    # Display the plots side by side
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
    
    st.write('Kurtosis of Months on book features is: {}'.format(bank['Months_on_book'].kurt()))
    st.write('We have a low kurtosis value pointing to a very flat shaped distribution (as shown in the plots above as well), meaning we cannot assume normality of the feature.\n')

# **Distribution of Total no. of products**

with tab1:
    st.markdown("<h3 style='text-align: center;'>\nDistribution of Total no. of products held by the customer</h3>", unsafe_allow_html=True)
    fig1 = go.Figure(go.Box(x=bank['Total_Relationship_Count'], name='Total no. of products Box plot', boxmean=True))
    fig2 = go.Figure(go.Histogram(x=bank['Total_Relationship_Count'], name='Total no. of products Histogram'))
    
    #Create two columns in streamlit
    col1, col2 = st.columns(2)
    
    # Display plots side by side
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        
    st.write('The distribution of the total number of products held by the customer seems closer to a uniform distribution and may appear useless as a predictor for churn status.')

# **Distribution of the number of months inactive in the last 12 months**

with tab1:
    st.markdown("<h3 style='text-align: center;'>\nDistribution of the number of months inactive in the last 12 months</h3>", unsafe_allow_html=True)
    fig1 = go.Figure(go.Box(x=bank['Months_Inactive_12_mon'], name='number of months inactive Box Plot', boxmean=True))
    fig2 = go.Figure(go.Histogram(x=bank['Months_Inactive_12_mon'], name='number of months inactive Histogram'))
    
    #Create two columns in streamlit
    col1, col2 = st.columns(2)
    
    # Display plots side by side
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


# **Distribution of the Credit Limit**

with tab1:
    st.markdown("<h3 style='text-align: center;'>\nDistribution of the Credit Limit</h3>", unsafe_allow_html=True)
    fig1 = go.Figure(go.Box(x=bank['Credit_Limit'], name='Credit_Limit Box Plot', boxmean=True))
    fig2 = go.Figure(go.Histogram(x=bank['Credit_Limit'], name='Credit_Limit Histogram'))
    
    #Create two columns in streamlit
    col1, col2 = st.columns(2)
    
    # Display plots side by side
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


# **Distribution of the Total Transaction Amount**

with tab1:
    st.markdown("<h3 style='text-align: center;'>\nDistribution of the Total Transaction Amount (Last 12 months)</h3>", unsafe_allow_html=True)
    fig1 = go.Figure(go.Box(x=bank['Total_Trans_Amt'], name='Total_Trans_Amt Box Plot', boxmean=True))
    fig2 = go.Figure(go.Histogram(x=bank['Total_Trans_Amt'], name='Total_Trans_Amt Histogram'))
    
    #Create two columns in streamlit
    col1, col2 = st.columns(2)
    
    # Display plots side by side
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        
    st.write('We see that the distribution of the total transactions (Last 12 months) displays a multimodal distribution, meaning we have some underlying groups in our data; it can be an interesting experiment to try and cluster the different groups and view the similarities between them and what describes best the different groups which create the different modes in our distribution.')

# **Proportion of churn vs not churn customers**

with tab1:
    st.markdown("<h3 style='text-align: center;'>\nProportion of churn vs not churn customers</h3>", unsafe_allow_html=True)
    fig1 = px.pie(bank, names='Attrition_Flag', hole=0.33)
    
    st.plotly_chart(fig1)
    st.write('As we can see, only 16% of the data samples represent churn customers; in the following steps, I will use SMOTE to upsample the churn samples to match them with the regular customer sample size to give the later selected models a better chance of catching on small details which will almost definitely be missed out with such a size difference.')

# ### Machine Learning
#
# **Data Preprocessing**

c_data = bank.copy()

with tab2:
    c_data
    c_data.Attrition_Flag = c_data.Attrition_Flag.replace({'Attrited Customer':1,'Existing Customer':0})
    c_data.Gender = c_data.Gender.replace({'F':1,'M':0})
    c_data = pd.concat([c_data,pd.get_dummies(c_data['Education_Level']).drop(columns=['Unknown'])],axis=1)
    c_data = pd.concat([c_data,pd.get_dummies(c_data['Income_Category']).drop(columns=['Unknown'])],axis=1)
    c_data = pd.concat([c_data,pd.get_dummies(c_data['Marital_Status']).drop(columns=['Unknown'])],axis=1)
    c_data = pd.concat([c_data,pd.get_dummies(c_data['Card_Category']).drop(columns=['Platinum'])],axis=1)
    c_data.drop(columns = ['Education_Level','Income_Category','Marital_Status','Card_Category','CLIENTNUM'],inplace=True)

# **Here we one hot encode all the categorical features describing different statuses of a customer.**

with tab2:
    fig = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=('Perason Correaltion',  'Spearman Correaltion'))
    colorscale=     [[1.0              , "rgb(165,0,38)"],
                    [0.8888888888888888, "rgb(215,48,39)"],
                    [0.7777777777777778, "rgb(244,109,67)"],
                    [0.6666666666666666, "rgb(253,174,97)"],
                    [0.5555555555555556, "rgb(254,224,144)"],
                    [0.4444444444444444, "rgb(224,243,248)"],
                    [0.3333333333333333, "rgb(171,217,233)"],
                    [0.2222222222222222, "rgb(116,173,209)"],
                    [0.1111111111111111, "rgb(69,117,180)"],
                    [0.0               , "rgb(49,54,149)"]]

    s_val =c_data.corr('pearson')
    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,name='pearson',showscale=False,xgap=0.7,ygap=0.7,colorscale=colorscale),
        row=1, col=1
    )


    s_val =c_data.corr('spearman')
    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=0.7,ygap=0.7,colorscale=colorscale),
        row=2, col=1
    )
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    
    fig.update_layout(height=1500, width=900)
    st.plotly_chart(fig, use_container_width=True)

# **Data Upsampling Using SMOTE**

with tab2:
    oversample = SMOTE()
    X, y = oversample.fit_resample(c_data[c_data.columns[1:]], c_data[c_data.columns[0]])
    usampled_df = X.assign(Churn = y)
    
    ohe_data =usampled_df[usampled_df.columns[15:-1]].copy()
    usampled_df = usampled_df.drop(columns=usampled_df.columns[15:-1])

with tab2:
    fig = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=('Perason Correaltion',  'Spearman Correaltion'))
    colorscale=     [[1.0              , "rgb(165,0,38)"],
                    [0.8888888888888888, "rgb(215,48,39)"],
                    [0.7777777777777778, "rgb(244,109,67)"],
                    [0.6666666666666666, "rgb(253,174,97)"],
                    [0.5555555555555556, "rgb(254,224,144)"],
                    [0.4444444444444444, "rgb(224,243,248)"],
                    [0.3333333333333333, "rgb(171,217,233)"],
                    [0.2222222222222222, "rgb(116,173,209)"],
                    [0.1111111111111111, "rgb(69,117,180)"],
                    [0.0               , "rgb(49,54,149)"]]

    s_val =usampled_df.corr('pearson')
    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,name='pearson',showscale=False,xgap=1,ygap=1,colorscale=colorscale),
        row=1, col=1
    )


    s_val =usampled_df.corr('spearman')
    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=1,ygap=1,colorscale=colorscale),
        row=2, col=1
    )
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )
    fig.update_layout(height=1500, width=900)
    st.plotly_chart(fig, use_container_width=True)

# **Principal Component Analysis Of One Hot Encoded Data**<br>
# We will use principal component analysis to reduce the dimensionality of the one-hot encoded categorical variables losing some of the variances, but simultaneously, using a couple of principal components instead of tens of one-hot encoded features will help me construct a better model.

with tab2:
    N_COMPONENTS = 4

    pca_model = PCA(n_components = N_COMPONENTS )

    pc_matrix = pca_model.fit_transform(ohe_data)

    evr = pca_model.explained_variance_ratio_
    total_var = evr.sum() * 100
    cumsum_evr = np.cumsum(evr)

    trace1 = {
        "name": "individual explained variance", 
        "type": "bar", 
        'y':evr}
    trace2 = {
        "name": "cumulative explained variance", 
        "type": "scatter", 
         'y':cumsum_evr}
    data = [trace1, trace2]
    layout = {
        "xaxis": {"title": "Principal components"}, 
        "yaxis": {"title": "Explained variance ratio"},
      }
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(     title='Explained Variance Using {} Dimensions'.format(N_COMPONENTS))
    st.plotly_chart(fig)

with tab2:
    usampled_df_with_pcs = pd.concat([usampled_df,pd.DataFrame(pc_matrix,columns=['PC-{}'.format(i) for i in range(0,N_COMPONENTS)])],axis=1)
    fig = px.scatter_matrix(
    usampled_df_with_pcs[['PC-{}'.format(i) for i in range(0,N_COMPONENTS)]].values,
    color=usampled_df_with_pcs.Credit_Limit,
    dimensions=range(N_COMPONENTS),
    labels={str(i):'PC-{}'.format(i) for i in range(0,N_COMPONENTS)},
    title=f'Total Explained Variance: {total_var:.2f}%')

    fig.update_traces(diagonal_visible=False)
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Credit_Limit",
        ),
    )
    
    st.plotly_chart(fig)

with tab2:
    fig = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=('Perason Correaltion',  'Spearman Correaltion'))

    s_val =usampled_df_with_pcs.corr('pearson')
    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,name='pearson',showscale=False,xgap=1,ygap=1,colorscale=colorscale),
        row=1, col=1
    )


    s_val =usampled_df_with_pcs.corr('spearman')
    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=1,ygap=1,colorscale=colorscale),
        row=2, col=1
    )
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    fig.update_layout(height=700,title_text="Upsmapled Correlations With PC\'s")
    st.plotly_chart(fig)

# **Model Selection And Evaluation**

with tab2:
    X_features = ['Total_Trans_Ct','PC-3','PC-1','PC-0','PC-2','Total_Ct_Chng_Q4_Q1','Total_Relationship_Count']

    X = usampled_df_with_pcs[X_features]
    y = usampled_df_with_pcs['Churn']
    
    train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=42)

# **Cross Validation**


with tab2:
    rf_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",RandomForestClassifier(random_state=42)) ])
    ada_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",AdaBoostClassifier(random_state=42,learning_rate=0.7)) ])
    svm_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",SVC(random_state=42,kernel='rbf')) ])


    f1_cross_val_scores = cross_val_score(rf_pipe,train_x,train_y,cv=5,scoring='f1')
    ada_f1_cross_val_scores=cross_val_score(ada_pipe,train_x,train_y,cv=5,scoring='f1')
    svm_f1_cross_val_scores=cross_val_score(svm_pipe,train_x,train_y,cv=5,scoring='f1')

with tab2:
    fig = make_subplots(rows=3, cols=1,shared_xaxes=True,subplot_titles=('Random Forest Cross Val Scores',
                                                                         'Adaboost Cross Val Scores',
                                                                        'SVM Cross Val Scores'))

    fig.add_trace(
        go.Scatter(x=list(range(0,len(f1_cross_val_scores))),y=f1_cross_val_scores,name='Random Forest'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(0,len(ada_f1_cross_val_scores))),y=ada_f1_cross_val_scores,name='Adaboost'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(0,len(svm_f1_cross_val_scores))),y=svm_f1_cross_val_scores,name='SVM'),
        row=3, col=1
    )

    fig.update_layout(height=700, title_text="Different Model 5 Fold Cross Validation")
    fig.update_yaxes(title_text="F1 Score")
    fig.update_xaxes(title_text="Fold #")
    
    st.image('newplot (1).png', caption='Model Comparison Summary')


# **Model Evaluation**

with tab2:
    rf_pipe.fit(train_x,train_y)
    rf_prediction = rf_pipe.predict(test_x)

    ada_pipe.fit(train_x,train_y)
    ada_prediction = ada_pipe.predict(test_x)

    svm_pipe.fit(train_x,train_y)
    svm_prediction = svm_pipe.predict(test_x)

with tab2:
    fig = go.Figure(data=[go.Table(header=dict(values=['<b>Model<b>', '<b>F1 Score On Test Data<b>'],
                                           line_color='darkslategray',
    fill_color='whitesmoke',
    align=['center','center'],
    font=dict(color='black', size=18),
    height=40),
                               
                 cells=dict(values=[['<b>Random Forest<b>', '<b>AdaBoost<b>','<b>SVM<b>'], [np.round(f1(rf_prediction,test_y),2), 
                                                                          np.round(f1(ada_prediction,test_y),2),
                                                                          np.round(f1(svm_prediction,test_y),2)]]))
                     ])

    fig.update_layout(title='Model Results On Test Data')
    st.image('newplot (2).png', caption='Model Results On Test Data')

# **Model Evaluation On Original Data (Before Upsampling)**

with tab2:
    ohe_data =c_data[c_data.columns[16:]].copy()
    pc_matrix = pca_model.fit_transform(ohe_data)
    original_df_with_pcs = pd.concat([c_data,pd.DataFrame(pc_matrix,columns=['PC-{}'.format(i) for i in range(0,N_COMPONENTS)])],axis=1)

    unsampled_data_prediction_RF = rf_pipe.predict(original_df_with_pcs[X_features])
    unsampled_data_prediction_ADA = ada_pipe.predict(original_df_with_pcs[X_features])
    unsampled_data_prediction_SVM = svm_pipe.predict(original_df_with_pcs[X_features])

# +
with tab2:
#     fig = go.Figure(data=[go.Table(header=dict(values=['<b>Model<b>', '<b>F1 Score On Original Data (Before Upsampling)<b>'],
#                                            line_color='darkslategray',
#     fill_color='whitesmoke',
#     align=['center','center'],
#     font=dict(color='black', size=18),
#     height=40),
                               
#                  cells=dict(values=[['<b>Random Forest<b>', '<b>AdaBoost<b>','<b>SVM<b>'], [np.round(f1(unsampled_data_prediction_RF,original_df_with_pcs['Attrition_Flag']),2), 
#                                                                           np.round(f1(unsampled_data_prediction_ADA,original_df_with_pcs['Attrition_Flag']),2),
#                                                                           np.round(f1(unsampled_data_prediction_SVM,original_df_with_pcs['Attrition_Flag']),2)]]))
#                      ])

#     fig.update_layout(title='Model Result On Original Data (Without Upsampling)')
    st.image('newplot (3).png', caption='Model Result On Original Data (Without Upsampling)')
# -

# **Results**

with tab2:
#     import plotly.figure_factory as ff
#     z=confusion_matrix(unsampled_data_prediction_RF,original_df_with_pcs['Attrition_Flag'])
#     fig = ff.create_annotated_heatmap(z, x=['Not Churn','Churn'], y=['Predicted Not Churn','Predicted Churn'], colorscale='Fall',xgap=3,ygap=3)
#     fig['data'][0]['showscale'] = True
#     fig.update_layout(title='Prediction On Original Data With Random Forest Model Confusion Matrix')
    st.image('newplot (4).png', caption='rediction On Original Data With Random Forest Model Confusion Matrix')

# +
with tab2:
#     from numpy import interp

#     import scikitplot as skplt
#     unsampled_data_prediction_RF = rf_pipe.predict_proba(original_df_with_pcs[X_features])
#     skplt.metrics.plot_precision_recall(original_df_with_pcs['Attrition_Flag'], unsampled_data_prediction_RF)
#     plt.legend(prop={'size': 20})
    st.image('newplot(5).png', caption='rediction On Original Data With Random Forest Model Confusion Matrix')
# -

# !streamlit run Bank_churn.py



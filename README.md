# Covid_LSTM_Predictions

## Introduccion ## 
    LSTM networks as known as good fit for timeseries, specially in those case where sudden changes can be expected. 
    Therefore this can represent a promigsing tool to make predictions over the covid_19 fast changing timeseries. 
    In this case the next day is predicted and shown in a temporary deployed website generated with dash. 
    The prediction is done for a country choosen by the user. 
    The prediction represent the cumulative sum of covid cases. 
    The number of days for the cumulative sum is chossen by the user. 
    The cumulative sum is used to avoid noise of the original data. 
    
## Scripts contained ##
### settings ###
     The user defines the user directory, country and number of cumulative days to be considered for the predictions. 
   
### Load_Data ###
     The data is loaded from public website database defined in the script. 
     The data is tunned to calculate the cumulative sum, 
     prepared for further analysis and storaged. 
   
### Pred_LSTM ###
     Data for the selected country is loaded and prepared to be trained by a LSTM network.
     LSTM network is generated using tensorflow and keras packages. 
     LSTM network is trained and prediction results are plotted. 
     Results are saved for further analysis. 
   
### Prod_WebApp ###
     Load_Data and Pred_LSTM scripts run at the begining of the script.
     Data is tunned and represented into a temporary website. 
     
## Folder configuration ##
     Create a Covid folder under use directory. 
     Save the scripts under the Covid folder. 
     Create a Data folder under the Covid directoy. 
     
## Run ## 
     Run the Prod_WebApp.py as it will call the other scripts. 
     The run will prompt some plots with predictions. Close them to continue with the web deployment. 
     The server will run over http://127.0.0.1:8050/
   
   

    
    

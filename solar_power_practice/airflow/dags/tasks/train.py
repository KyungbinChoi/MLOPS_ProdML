#!/usr/bin/env python
# coding: utf-8

def train():
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import plotly.graph_objects as go
    from datetime import datetime
    import pytz

    kst = pytz.timezone('Asia/Seoul')
    now=datetime.now(kst)

    generation1 = pd.read_csv("dags/datasets/train/Plant_1_Generation_Data.csv")
    weather1 = pd.read_csv("dags/datasets/train/Plant_1_Weather_Sensor_Data.csv")
    generation1['DATE_TIME'] = pd.to_datetime(generation1['DATE_TIME'], dayfirst=True)
    weather1['DATE_TIME'] = pd.to_datetime(weather1['DATE_TIME'], dayfirst=False)


    # In[3]:


    generation1


    # In[4]:


    inverters = list(generation1['SOURCE_KEY'].unique())
    print(f"total number of inverters {len(inverters)}")


    # # Inverter level Anomally detection

    # In[5]:


    inverters[0]


    # In[6]:


    inv_1 = generation1[generation1['SOURCE_KEY']==inverters[0]]
    mask = ((weather1['DATE_TIME'] >= min(inv_1["DATE_TIME"])) & (weather1['DATE_TIME'] <= max(inv_1["DATE_TIME"])))
    weather_filtered = weather1.loc[mask]


    # In[7]:


    weather_filtered.shape


    # In[8]:


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=inv_1["DATE_TIME"], y=inv_1["AC_POWER"],
                        mode='lines',
                        name='AC Power'))

    fig.add_trace(go.Scatter(x=weather_filtered["DATE_TIME"], y=weather_filtered["IRRADIATION"],
                        mode='lines',
                        name='Irradiation', 
                        yaxis='y2'))

    fig.update_layout(title_text="Irradiation vs AC POWER",
                    yaxis1=dict(title="AC Power in kW",
                                side='left'),
                    yaxis2=dict(title="Irradiation index",
                                side='right',
                                anchor="x",
                                overlaying="y"
                                ))

    fig.write_image(f"dags/outputs/train/{now}_AC_power.png")


    # ### Graph observations
    # We can see that in June 7th and June 14th there are some misproduction areas that could be considered anomalies. Due to the fact that energy production should behave in a linear way to irradiation.

    # In[9]:


    df = inv_1.merge(weather_filtered, on="DATE_TIME", how='left')
    df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
    df


    # # LSTM Autoencoder approach

    # In[10]:


    df = df[["DATE_TIME", "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    df_timestamp = df[["DATE_TIME"]]
    df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]


    # In[11]:


    train_prp = .6
    train = df_.loc[:df_.shape[0]*train_prp]
    test = df_.loc[df_.shape[0]*train_prp:]


    # In[12]:


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train)
    X_test = scaler.transform(test)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")


    # In[13]:


    from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers


    # In[14]:


    def autoencoder_model(X):
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(X.shape[2]))(L5)
        model = Model(inputs=inputs, outputs=output)
        return model


    # In[15]:


    model = autoencoder_model(X_train)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()


    # In[16]:


    epochs = 100
    batch = 10
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch, validation_split=.2, verbose=0).history
    model.save("dags/models/lstm.h5")


    # In[17]:


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[x for x in range(len(history['loss']))], y=history['loss'],
                        mode='lines',
                        name='loss'))

    fig.add_trace(go.Scatter(x=[x for x in range(len(history['val_loss']))], y=history['val_loss'],
                        mode='lines',
                        name='validation loss'))

    fig.update_layout(title="Autoencoder error loss over epochs",
                    yaxis=dict(title="Loss"),
                    xaxis=dict(title="Epoch"))

    fig.write_image(f"dags/outputs/train/{now}_Error_Loss.png")


    # In[18]:


    X_pred = model.predict(X_train)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = scaler.inverse_transform(X_pred)
    X_pred = pd.DataFrame(X_pred, columns=train.columns)


    # In[19]:


    scores = pd.DataFrame()
    scores['AC_train'] = train['AC_POWER']
    scores["AC_predicted"] = X_pred["AC_POWER"]
    scores['loss_mae'] = (scores['AC_train']-scores['AC_predicted']).abs()


    # In[20]:


    fig = go.Figure(data=[go.Histogram(x=scores['loss_mae'])])
    fig.update_layout(title="Error distribution", 
                    xaxis=dict(title="Error delta between predicted and real data [AC Power]"),
                    yaxis=dict(title="Data point counts"))

    fig.write_image(f"dags/outputs/train/{now}_Error_Distribution.png")


    # In[21]:


    X_pred = model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = scaler.inverse_transform(X_pred)
    X_pred = pd.DataFrame(X_pred, columns=train.columns)
    X_pred.index = test.index


    # In[22]:


    scores = X_pred
    scores['datetime'] = df_timestamp.loc[1893:]
    scores['real AC'] = test['AC_POWER']
    scores["loss_mae"] = (scores['real AC'] - scores['AC_POWER']).abs()
    scores['Threshold'] = 200
    scores['Anomaly'] = np.where(scores["loss_mae"] > scores["Threshold"], 1, 0)


    # In[23]:


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores['datetime'], 
                            y=scores['loss_mae'], 
                            name="Loss"))
    fig.add_trace(go.Scatter(x=scores['datetime'], 
                            y=scores['Threshold'],
                            name="Threshold"))

    fig.update_layout(title="Error Timeseries and Threshold", 
                    xaxis=dict(title="DateTime"),
                    yaxis=dict(title="Loss"))
    fig.write_image(f"dags/outputs/train/{now}_Threshold.png")


    # In[24]:


    scores['Anomaly'].value_counts()


    # In[25]:


    anomalies = scores[scores['Anomaly'] == 1][['real AC']]
    anomalies = anomalies.rename(columns={'real AC':'anomalies'})
    scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')


    # In[26]:


    scores[(scores['Anomaly'] == 1) & (scores['datetime'].notnull())].to_csv(f"dags/outputs/train/{now}_anomalies.csv", index=False)


    # In[27]:


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["real AC"],
                        mode='lines',
                        name='AC Power'))

    fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["anomalies"],
                        name='Anomaly', 
                        mode='markers',
                        marker=dict(color="red",
                                    size=11,
                                    line=dict(color="red",
                                            width=2))))

    fig.update_layout(title_text="Anomalies Detected LSTM Autoencoder")

    fig.write_image(f"dags/outputs/train/{now}_Anomaly.png")



if __name__ == "__main__":
    train()
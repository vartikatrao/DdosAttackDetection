import pickle
from flask import Flask, render_template
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (for PNGs) instead of the default interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import mean_squared_error
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
app = Flask(__name__,template_folder='Templates')
from flask import request
import time 
import joblib 

@app.route('/kaggle_balanced')
def kaggle_balanced():
    return render_template('kaggle_balanced.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/kaggle_unbalanced')
def kaggle_unbalanced():
    return render_template('kaggle_unbalanced.html')

@app.route('/CICIDS_1')
def CICIDS_1():
    return render_template('CICIDS_1.html')

@app.route('/CICIDS_2')
def CICIDS_2():
    return render_template('CICIDS_2.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/selectdataset')
def selectdataset():
    return render_template('selectdataset.html')

model_cnn_balanced = keras.models.load_model('models//kaggle//balanced//cnn_model.h5')
model_dnn_balanced= keras.models.load_model('models//kaggle//balanced//dnn_model.h5')
@app.route('/predict_balanced', methods=['GET', 'POST'])
def predict_balanced():
    if request.method == 'POST':

        # Load the uploaded files
        file = request.files['fileInput'] 
        dtypes = {
    'Src IP': 'category',
    'Src Port': 'uint16',
    'Dst IP': 'category',
    'Dst Port': 'uint16',
    'Protocol': 'category',
    'Flow Duration': 'uint32',
    'Tot Fwd Pkts': 'uint32',
    'Tot Bwd Pkts': 'uint32',
    'TotLen Fwd Pkts': 'float32',
    'TotLen Bwd Pkts': 'float32',
    'Fwd Pkt Len Max': 'float32',
    'Fwd Pkt Len Min': 'float32',
    'Fwd Pkt Len Mean': 'float32',
    'Fwd Pkt Len Std': 'float32',
    'Bwd Pkt Len Max': 'float32',
    'Bwd Pkt Len Min': 'float32',
    'Bwd Pkt Len Mean': 'float32',
    'Bwd Pkt Len Std': 'float32',
    'Flow Byts/s': 'float32',
    'Flow Pkts/s': 'float32',
    'Flow IAT Mean': 'float32',
    'Flow IAT Std': 'float32',
    'Flow IAT Max': 'float32',
    'Flow IAT Min': 'float32',
    'Fwd IAT Tot': 'float32',
    'Fwd IAT Mean': 'float32',
    'Fwd IAT Std': 'float32',
    'Fwd IAT Max': 'float32',
    'Fwd IAT Min': 'float32',
    'Bwd IAT Tot': 'float32',
    'Bwd IAT Mean': 'float32',
    'Bwd IAT Std': 'float32',
    'Bwd IAT Max': 'float32',
    'Bwd IAT Min': 'float32',
    'Fwd PSH Flags': 'category',
    'Bwd PSH Flags': 'category',
    'Fwd URG Flags': 'category',
    'Bwd URG Flags': 'category',
    'Fwd Header Len': 'uint32',
    'Bwd Header Len': 'uint32',
    'Fwd Pkts/s': 'float32',
    'Bwd Pkts/s': 'float32',
    'Pkt Len Min': 'float32',
    'Pkt Len Max': 'float32',
    'Pkt Len Mean': 'float32',
    'Pkt Len Std': 'float32',
    'Pkt Len Var': 'float32',
    'FIN Flag Cnt': 'category',
    'SYN Flag Cnt': 'category',
    'RST Flag Cnt': 'category',
    'PSH Flag Cnt': 'category',
    'ACK Flag Cnt': 'category',
    'URG Flag Cnt': 'category',
    'CWE Flag Count': 'category',
    'ECE Flag Cnt': 'category',
    'Down/Up Ratio': 'float32',
    'Pkt Size Avg': 'float32',
    'Fwd Seg Size Avg': 'float32',
    'Bwd Seg Size Avg': 'float32',
    'Fwd Byts/b Avg': 'uint32',
    'Fwd Pkts/b Avg': 'uint32',
    'Fwd Blk Rate Avg': 'uint32',
    'Bwd Byts/b Avg': 'uint32',
    'Bwd Pkts/b Avg': 'uint32',
    'Bwd Blk Rate Avg': 'uint32',
    'Subflow Fwd Pkts': 'uint32',
    'Subflow Fwd Byts': 'uint32',
    'Subflow Bwd Pkts': 'uint32',
    'Subflow Bwd Byts': 'uint32',
    'Init Fwd Win Byts': 'uint32',
    'Init Bwd Win Byts': 'uint32',
    'Fwd Act Data Pkts': 'uint32',
    'Fwd Seg Size Min': 'uint32',
    'Active Mean': 'float32',
    'Active Std': 'float32',
    'Active Max': 'float32',
    'Active Min': 'float32',
    'Idle Mean': 'float32',
    'Idle Std': 'float32',
    'Idle Max': 'float32',
    'Idle Min': 'float32',
    'Label': 'category'
}
        df = pd.read_csv(file, dtype=dtypes)

        # List of columns to keep
        desired_columns = ['Src Port', 'Dst Port', 'Protocol', 'Flow Duration',
                        'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Fwd Pkt Len Max',
                        'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Max',
                        'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Flow Byts/s',
                        'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                        'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std',
                        'Fwd IAT Max', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std',
                        'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Pkts/s', 'Bwd Pkts/s',
                        'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
                        'Pkt Len Var', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
                        'ACK Flag Cnt', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
                        'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Seg Size Min',
                        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
                        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
        df = df[desired_columns]
        # Keep only the desired columns
        x_test = df.values
        x_test_cnn = x_test.reshape(x_test.shape[0], 52, 1).astype('float32')

        start_time_cnn = time.time()
        x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_pred_cnn= np.argmax(model_dnn_balanced.predict(x_test_tensor), axis=-1)
        end_time_cnn= time.time()
        inference_time_cnn= end_time_cnn- start_time_cnn
        print(y_pred_cnn)
        x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
        start_time_dnn= time.time()
        y_pred_dnn= np.argmax(model_dnn_balanced.predict(x_test_tensor), axis=-1)
        end_time_dnn= time.time()
        inference_time_dnn= end_time_dnn- start_time_dnn
        print(y_pred_dnn)
        models = ['DCNN', 'DNN']
        inference_times = [inference_time_cnn, inference_time_dnn]
         # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a scatter plot of the predicted and expected labels
        ax.bar(models[::-1], inference_times[::-1], color=['purple', 'coral'])

        # Set the x and y axis labels and the title
        ax.set_xlabel('Models', fontsize=20, color="white")
        ax.set_ylabel('Inference Times ', fontsize=20, color='white')
        ax.set_title('Models vs Inference Times', fontsize=28, color= 'white')

        # Set the x and y axis ticks color and size
        ax.tick_params(axis='x', colors='white', labelsize=16)
        ax.tick_params(axis='y', colors='white', labelsize=16)

        # Remove the gridlines and add a legend
        ax.grid(False)
        ax.legend()
        fig.patch.set_alpha(0)
        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        # Convert the buffer content to a string and encode it as base64
        data = base64.b64encode(buf.getbuffer()).decode('ascii')

        pred_dict = {0: 'ddos', 1: 'safe'}
        cnn_prediction = pred_dict[y_pred_cnn[0]]
        dnn_prediction = pred_dict[y_pred_dnn[0]]
        return render_template('results.html', cnn_prediction=cnn_prediction, dnn_prediction=dnn_prediction, plot_data= data, cnn_inference_time=inference_time_cnn, dnn_inference_time= inference_time_dnn )
    else:
        return render_template('results.html')


model_cnn_unbalanced = keras.models.load_model('models//kaggle//unbalanced//dnn_model.h5')
model_dnn_unbalanced= keras.models.load_model('models//kaggle//unbalanced//cnn_model.h5')
@app.route('/predict_unbalanced', methods=['GET', 'POST'])
def predict_unbalanced():
    if request.method == 'POST':

        # Load the uploaded files
        file = request.files['fileInput'] 
        dtypes = {
            'Src IP': 'category',
            'Src Port': 'uint16',
            'Dst IP': 'category',
            'Dst Port': 'uint16',
            'Protocol': 'category',
            'Flow Duration': 'uint32',
            'Tot Fwd Pkts': 'uint32',
            'Tot Bwd Pkts': 'uint32',
            'TotLen Fwd Pkts': 'float32',
            'TotLen Bwd Pkts': 'float32',
            'Fwd Pkt Len Max': 'float32',
            'Fwd Pkt Len Min': 'float32',
            'Fwd Pkt Len Mean': 'float32',
            'Fwd Pkt Len Std': 'float32',
            'Bwd Pkt Len Max': 'float32',
            'Bwd Pkt Len Min': 'float32',
            'Bwd Pkt Len Mean': 'float32',
            'Bwd Pkt Len Std': 'float32',
            'Flow Byts/s': 'float32',
            'Flow Pkts/s': 'float32',
            'Flow IAT Mean': 'float32',
            'Flow IAT Std': 'float32',
            'Flow IAT Max': 'float32',
            'Flow IAT Min': 'float32',
            'Fwd IAT Tot': 'float32',
            'Fwd IAT Mean': 'float32',
            'Fwd IAT Std': 'float32',
            'Fwd IAT Max': 'float32',
            'Fwd IAT Min': 'float32',
            'Bwd IAT Tot': 'float32',
            'Bwd IAT Mean': 'float32',
            'Bwd IAT Std': 'float32',
            'Bwd IAT Max': 'float32',
            'Bwd IAT Min': 'float32',
            'Fwd PSH Flags': 'category',
            'Bwd PSH Flags': 'category',
            'Fwd URG Flags': 'category',
            'Bwd URG Flags': 'category',
            'Fwd Header Len': 'uint32',
            'Bwd Header Len': 'uint32',
            'Fwd Pkts/s': 'float32',
            'Bwd Pkts/s': 'float32',
            'Pkt Len Min': 'float32',
            'Pkt Len Max': 'float32',
            'Pkt Len Mean': 'float32',
            'Pkt Len Std': 'float32',
            'Pkt Len Var': 'float32',
            'FIN Flag Cnt': 'category',
            'SYN Flag Cnt': 'category',
            'RST Flag Cnt': 'category',
            'PSH Flag Cnt': 'category',
            'ACK Flag Cnt': 'category',
            'URG Flag Cnt': 'category',
            'CWE Flag Count': 'category',
            'ECE Flag Cnt': 'category',
            'Down/Up Ratio': 'float32',
            'Pkt Size Avg': 'float32',
            'Fwd Seg Size Avg': 'float32',
            'Bwd Seg Size Avg': 'float32',
            'Fwd Byts/b Avg': 'uint32',
            'Fwd Pkts/b Avg': 'uint32',
            'Fwd Blk Rate Avg': 'uint32',
            'Bwd Byts/b Avg': 'uint32',
            'Bwd Pkts/b Avg': 'uint32',
            'Bwd Blk Rate Avg': 'uint32',
            'Subflow Fwd Pkts': 'uint32',
            'Subflow Fwd Byts': 'uint32',
            'Subflow Bwd Pkts': 'uint32',
            'Subflow Bwd Byts': 'uint32',
            'Init Fwd Win Byts': 'uint32',
            'Init Bwd Win Byts': 'uint32',
            'Fwd Act Data Pkts': 'uint32',
            'Fwd Seg Size Min': 'uint32',
            'Active Mean': 'float32',
            'Active Std': 'float32',
            'Active Max': 'float32',
            'Active Min': 'float32',
            'Idle Mean': 'float32',
            'Idle Std': 'float32',
            'Idle Max': 'float32',
            'Idle Min': 'float32',
            'Label': 'category'
        }
        df = pd.read_csv(file, dtype=dtypes)

        desired_columns = ['Src Port', 'Dst Port', 'Protocol', 'Flow Duration',
                            'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Fwd Pkt Len Max',
                            'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Max',
                            'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Flow Byts/s',
                            'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                            'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std',
                            'Fwd IAT Max', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std',
                            'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Pkts/s', 'Bwd Pkts/s',
                            'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
                            'Pkt Len Var', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
                            'ACK Flag Cnt', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
                            'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Seg Size Min',
                            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
                            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
        df = df[desired_columns]
        x_test = df.values
        x_test_cnn = x_test.reshape(x_test.shape[0], 52, 1).astype('float32')
        x_test_cnn =  tf.convert_to_tensor(x_test, dtype=tf.float32)
        start_time_cnn = time.time()
        y_pred_cnn = np.argmax(model_cnn_unbalanced.predict(x_test_cnn), axis=-1)
        end_time_cnn = time.time()
        inference_time_cnn = end_time_cnn - start_time_cnn

        x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)

        start_time_dnn = time.time()
        y_pred_dnn = np.argmax(model_dnn_balanced.predict(x_test_tensor), axis=-1)
        end_time_dnn = time.time()
        inference_time_dnn = end_time_dnn - start_time_dnn

        models = ['DCNN', 'DNN']
        inference_times = [inference_time_cnn, inference_time_dnn]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(models[::-1], inference_times[::-1], color=['purple', 'coral'])
        ax.set_xlabel('Models', fontsize=20, color="white")
        ax.set_ylabel('Inference Times ', fontsize=20, color='white')
        ax.set_title('Models vs Inference Times', fontsize=28, color= 'white')
        ax.tick_params(axis='x', colors='white', labelsize=16)
        ax.tick_params(axis='y', colors='white', labelsize=16)
        ax.grid(False)
        ax.legend()
        fig.patch.set_alpha(0)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        data = base64.b64encode(buf.getbuffer()).decode('ascii')

        pred_dict = {0: 'ddos', 1: 'safe'}
        cnn_prediction = pred_dict[y_pred_cnn[0]]
        dnn_prediction = pred_dict[y_pred_dnn[0]]
        return render_template('results.html', cnn_prediction=cnn_prediction, dnn_prediction=dnn_prediction, plot_data= data, cnn_inference_time=inference_time_cnn, dnn_inference_time= inference_time_dnn )
def preprocess_cicids17_model(df):
    df = df.dropna().reset_index()
    dp=df['Destination Port']
    df=df.drop('Destination Port',axis=1)
    data_clean = df.dropna().reset_index()
    data_np = data_clean.to_numpy(dtype="float32")
    data_np = data_np[~np.isinf(data_np).any(axis=1)]
    X = data_np[:, 0:78]
    print(df['Label'])
    scaler = joblib.load('models//CICIDS_1//scaler.pkl')
    scaler_dcnn=joblib.load('models//CICIDS_1//scaler_dcnn.pkl')
    X_scaled = scaler.transform(data_np)
    X_scal_dcnn=scaler_dcnn.transform(data_np)
    return X_scaled,X_scal_dcnn
cicids1dcnn=keras.models.load_model('models//CICIDS_1//dcnn_model.h5')
cicids1dnn=keras.models.load_model('models//CICIDS_1//dnn_model.h5')
@app.route('/predict_CICIDS1', methods=['GET', 'POST'])
def predict_CICIDS1():
    if request.method == 'POST':
        file = request.files['fileInput']
        df = pd.read_csv(file)
        X,X1= preprocess_cicids17_model(df)
        start_time_cnn = time.time()
        pred_cnn = cicids1dcnn.predict(X1)
        predc = np.argmax(pred_cnn,axis=1)
        #print(predc)
        end_time_cnn = time.time()
        inference_time_cnn = end_time_cnn - start_time_cnn
        
        start_time_dnn = time.time()
        pred_dnn = cicids1dnn.predict(X)
        predd=np.argmax(pred_dnn,axis=1)
        print(predd)
        end_time_dnn = time.time()
        inference_time_dnn = end_time_dnn - start_time_dnn
        
        classes = ["Class {}".format(i) for i in range(15)]
        print(classes)
        print(pred_cnn)
        print(pred_dnn)
        attack_types_cnn = [classes[np.argmax(p)] for p in pred_cnn]
        attack_types_dnn = [classes[np.argmax(p)] for p in pred_dnn]
        label_mappings_dict = {
            0: 'BENIGN',
            1: 'Bot',
            2: 'DDoS',
            3: 'DoS GoldenEye',
            4: 'DoS Hulk',
            5: 'DoS Slowhttptest',
            6: 'DoS slowloris',
            7: 'FTP-Patator',
            8: 'Heartbleed',
            9: 'Infiltration',
            10: 'PortScan',
            11: 'SSH-Patator',
            12: 'Web Attack ï¿½ Brute Force',
            13: 'Web Attack ï¿½ Sql Injection',
            14: 'Web Attack ï¿½ XSS'
        }

        cnn = [label_mappings_dict[pred] for pred in predc]
        dnn = [label_mappings_dict[pred] for pred in predd]

        print("CNN Predictions:", cnn)
        print("DNN Predictions:", dnn)

        print(attack_types_cnn)
        print(attack_types_dnn)
        models = ['DCNN', 'DNN']
        inference_times = [inference_time_cnn, inference_time_dnn]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(models[::-1], inference_times[::-1], color=['purple', 'coral'])
        ax.set_xlabel('Models', fontsize=20, color="white")
        ax.set_ylabel('Inference Times ', fontsize=20, color='white')
        ax.set_title('Models vs Inference Times', fontsize=28, color= 'white')
        ax.tick_params(axis='x', colors='white', labelsize=16)
        ax.tick_params(axis='y', colors='white', labelsize=16)
        ax.grid(False)
        fig.patch.set_alpha(0)
        
        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        cnn= cnn[0]
        dnn= dnn[0]
        # Convert the buffer content to a string and encode it as base64
        data = base64.b64encode(buf.getbuffer()).decode('ascii')
        # Render the template with the results
        return render_template('results.html', 
                               cnn_prediction=cnn, 
                               dnn_prediction=dnn, 
                               plot_data=data, 
                               cnn_inference_time=inference_time_cnn, 
                               dnn_inference_time=inference_time_dnn)
    else:
        return render_template('results.html')

cicids2dcnn=keras.models.load_model('models//CICIDS_2//dcnn_model.h5')
cicids2dnn=keras.models.load_model('models//CICIDS_2//dnn_model.h5')
@app.route('/predict_CICIDS2', methods=['GET', 'POST'])
def predict_CICIDS2():
    if request.method == 'POST':
        file = request.files['fileInput']
        df = pd.read_csv(file)
        X,X1= preprocess_cicids17_model(df)
        start_time_cnn = time.time()
        pred_cnn = cicids1dcnn.predict(X1)
        predc = np.argmax(pred_cnn,axis=1)
        #print(predc)
        end_time_cnn = time.time()
        inference_time_cnn = end_time_cnn - start_time_cnn
        
        start_time_dnn = time.time()
        pred_dnn = cicids2dnn.predict(X)
        predd=np.argmax(pred_dnn,axis=1)
        print(predd)
        end_time_dnn = time.time()
        inference_time_dnn = end_time_dnn - start_time_dnn
        
        classes = ["Class {}".format(i) for i in range(15)]
        print(classes)
        print(pred_cnn)
        print(pred_dnn)
        attack_types_cnn = [classes[np.argmax(p)] for p in pred_cnn]
        attack_types_dnn = [classes[np.argmax(p)] for p in pred_dnn]
        label_mappings_dict = {
            0: 'BENIGN',
            1: 'Bot',
            2: 'DDoS',
            3: 'DoS GoldenEye',
            4: 'DoS Hulk',
            5: 'DoS Slowhttptest',
            6: 'DoS slowloris',
            7: 'FTP-Patator',
            8: 'Heartbleed',
            9: 'Infiltration',
            10: 'PortScan',
            11: 'SSH-Patator',
            12: 'Web Attack ï¿½ Brute Force',
            13: 'Web Attack ï¿½ Sql Injection',
            14: 'Web Attack ï¿½ XSS'
        }

        cnn = [label_mappings_dict[pred] for pred in predc]
        dnn = [label_mappings_dict[pred] for pred in predd]

        print("CNN Predictions:", cnn)
        print("DNN Predictions:", dnn)

        print(attack_types_cnn)
        print(attack_types_dnn)
        models = ['DCNN', 'DNN']
        inference_times = [inference_time_cnn, inference_time_dnn]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(models[::-1], inference_times[::-1], color=['purple', 'coral'])
        ax.set_xlabel('Models', fontsize=20, color="white")
        ax.set_ylabel('Inference Times ', fontsize=20, color='white')
        ax.set_title('Models vs Inference Times', fontsize=28, color= 'white')
        ax.tick_params(axis='x', colors='white', labelsize=16)
        ax.tick_params(axis='y', colors='white', labelsize=16)
        ax.grid(False)
        fig.patch.set_alpha(0)
        
        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        cnn= cnn[0]
        dnn= dnn[0]
        # Convert the buffer content to a string and encode it as base64
        data = base64.b64encode(buf.getbuffer()).decode('ascii')
        # Render the template with the results
        return render_template('results.html', 
                               cnn_prediction=cnn, 
                               dnn_prediction=dnn, 
                               plot_data=data, 
                               cnn_inference_time=inference_time_cnn, 
                               dnn_inference_time=inference_time_dnn)
    else:
        return render_template('results.html')


if __name__ == '__main__':
    app.run(debug=True)

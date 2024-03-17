import pickle
from flask import Flask, render_template
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import mean_squared_error
import pandas as pd  
app = Flask(__name__,template_folder='Templates')
#model = keras.models.load_model('lstm model\hi.h5')

from flask import request

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
    return render_template('CICIDS_1')

@app.route('/CICIDS_2')
def CICIDS_2():
    return render_template('CICIDS_2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/selectdataset')
def selectdataset():
    return render_template('selectdataset.html')

# @app.route('/predictlstm')
# def predictlstm():
#     return render_template('predictlstm.html')


# @app.route('/predictlstm', methods=['GET', 'POST'])
# def predict_lstm():
#     if request.method == 'POST':
#         # Load the uploaded files
#         file1 = request.files['file1']
#         file2 = request.files['file2']
        
#         # Load the numpy arrays from the uploaded files
#         xtest = np.load(file1)
#         ytest = np.load(file2)

#         # Use the model to make predictions on the test data
#         # y_pred = model.predict(xtest)

#         # Convert probabilities to binary predictions (0 or 1)
#         y_pred_binary = (y_pred > 0.5).astype(int)
#         accuracy = accuracy_score(ytest, y_pred_binary)
#         recall = recall_score(ytest, y_pred_binary)

#         # Create a figure and axis object
#         fig, ax = plt.subplots(figsize=(10, 8))

#         # Create a scatter plot of the predicted and expected labels
#         ax.scatter(range(len(ytest)), ytest, c='red', label='Expected Labels',s=150)
#         ax.scatter(range(len(y_pred_binary)), y_pred_binary, c='blue', label='Predicted Labels',s=50)

#         # Set the x and y axis labels and the title
#         ax.set_xlabel('Sample Number', fontsize=14)
#         ax.set_ylabel('Label', fontsize=14)
#         ax.set_title('Expected vs. Predicted Labels', fontsize=16)

#         # Remove the gridlines and add a legend
#         ax.grid(False)
#         ax.legend()

#         # Save the plot to a buffer
#         buf = BytesIO()
#         plt.savefig(buf, format='png')
#         plt.close()

#         # Convert the buffer content to a string and encode it as base64
#         data = base64.b64encode(buf.getbuffer()).decode('ascii')

#         # Render the afterlstm.html template and pass the predicted and expected labels and the plot data
#         return render_template('afterlstm.html', predicted_labels=y_pred_binary, expected_labels=ytest, accuracy=accuracy, recall=recall, plot_data=data)
#     else:
#         # Render the predictlstm.html template for GET requests
#         return render_template('predictlstm.html')

model_cnn_unbalanced = keras.models.load_model('models\\kaggle\\unbalanced\\unbalanced_cnn.h5')
model_dnn_unbalanced= keras.models.load_model('models\\kaggle\\unbalanced\\imbal_dnn_model.h5')
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
        y_pred_cnn= np.argmax(model_cnn_unbalanced.predict(x_test_cnn), axis=-1)
        print(y_pred_cnn)
        x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_pred_dnn= np.argmax(model_dnn_unbalanced.predict(x_test_tensor), axis=-1)
        print(y_pred_dnn)
        pred_dict= {0: 'ddos', 1: 'safe'}
        cnn_prediction= pred_dict[y_pred_cnn[0]]
        dnn_prediction= pred_dict[y_pred_dnn[0]]
        return render_template('results.html', cnn_prediction= cnn_prediction, dnn_prediction=dnn_prediction )
    else:
        # Render the predictlstm.html template for GET requests
        return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)

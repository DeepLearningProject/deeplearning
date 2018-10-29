### Deeplearning model for text classification

 - We have used the deep learning architecture used in the paper `Recurrent Convolutional Neural Networks for Text Classification` by Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552

 - Architecture suggested in the paper has an embedding layer, followed by bi-directional recurrent network and max-pooling layer. As suggested in the paper, from the perspective of convolutional neural networks, the recurrent structure we previously mentioned is the convolutional layer. When all of the representations of words are calculated, we apply a max-pooling layer. Tanh is the activation function and softmax is used for output layer

 - Training data was taken from Kaggle https://www.kaggle.com/c/twitter-sentiment-analysis2/data

 - File `LeadIQ Model Training.ipynb` in the repo represents Model training file, written in jupyter notebook
 
 - Model accuracy is very low at 59.95%. I couldn't run more epochs and increase batch size due to lack of computing power. My laptop has 8 GB RAM and no GPU. Everything it predicts is 1.
 
 - `model.h5` and `model.json` represent model weights and architecture respectively. `tokenizer.pickle` is the tokenizer trained on the data and will be used in the Flask API saved in the `FlaskAPI.py` file.
 
### To Launch API
 
  - Pre-requisites: This is developed in Windows 10 machine and will need `Python 3` and the python libraries `flask`, `numpy`, `keras`, `tensorflow`, `pickle` and `logging`.
 - download `model.h5`, `model.json`, `tokenizer.pickle` and `FlaskAPI.py`. Save it in a single folder.
 - edit `FlaskAPI.py` and provide the path in the path variable. replace actual path with the path `'G:/Script/LeadIQ'` in the line `path = 'G:/Script/LeadIQ'`


 - In the command line in windows, navigate to folder where `FlaskAPI.py` is saved locally. If you both Python2 and Python3, specify the version of python in 3 by typing `py -3.5 FlaskAPI.py` for example if your python is 3.5. Please note that this will not work in Python2. If you only have Python3, then simply type `python FlaskAPI.py` in the console to launch the API.
 
 - API is available at http://localhost:5000/predict

### Examples
For the sentence "I am happy", below is a worked out example in Python.

`import requests`

`example_text = 'I am happy'`

`requests.post('''http://localhost:5000/predict?text='''+example_text)`

output for this will be `<Response [200]>`

To see the exact output, use the text attribute.

`requests.post('''http://localhost:5000/predict?text='''+example_text).text` and the output will be 

`'{"prediction":"[1.]"}\n'`, which is positive. Since accuracy of the model is very low, most of the results are coming as 1.

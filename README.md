# Fashion-MNIST Classifier

This project is a web application built using Streamlit that classifies images of fashion items into one of the 10 categories from the Fashion-MNIST dataset. The app allows users to upload an image, preprocess it, and predict its class using a pre-trained machine learning model.

## Deplyment Link:
[Streamlit Deployment](https://recenttrendinai-js2uj2wu5eshotbthtvqqr.streamlit.app/)

** Only Valid for 7 days after deployment, as per free plan.

## Features
- Upload an image file (PNG, JPG, JPEG).
- Preprocess the image (grayscale, resize, normalize).
- Predict the class of the uploaded image using the best-trained model.
- Display the original and transformed images.
- Show a loading spinner while processing the image.

## Requirements
To run this application, you need the following dependencies:

- streamlit
- numpy
- pillow
- scikit-learn
- opencv-python
- tensorflow

Install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository or download the project files.
2. Ensure you have Python installed on your system.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
5. Open the provided URL in your browser to access the application.

## File Structure
- `app.py`: The main Streamlit application file.
- `best_model.pkl`: The pre-trained machine learning model.
- `requirements.txt`: List of dependencies for the project.

## How It Works
1. The user uploads an image file.
2. The app preprocesses the image (grayscale, resize to 28x28, normalize).
3. The pre-trained model predicts the class of the image.
4. The app displays the prediction along with the original and transformed images.

## Dataset
The model is trained on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), which contains 10 classes:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Machine Learning Model

The machine learning model was trained using the Fashion-MNIST dataset. The following supervised learning models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### Training Details
- **Dataset**: Fashion-MNIST
- **Preprocessing**: Images were flattened, normalized, and standardized.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, and AUC.
- **Best Model**: The model with the highest accuracy was selected as the best model and saved as `best_model.pkl`.

### Results
The results of the model evaluation are as follows:

| Model                | Accuracy | Precision | Recall | F1 Score | AUC   |
|----------------------|----------|-----------|--------|----------|-------|
| Logistic Regression  | 0.7972   | 0.797849  | 0.7972 | 0.797288 | 0.970087 |
| Decision Tree        | 0.7481   | 0.749920  | 0.7481 | 0.748898 | 0.860056 |
| Random Forest        | 0.8505   | 0.850107  | 0.8505 | 0.848844 | 0.985586 |
| SVM                  | 0.8511   | 0.850335  | 0.8511 | 0.850264 | 0.986410 |

## Additional Resources

- **Model Weights and Sample Images**: [Google Drive Link](https://drive.google.com/drive/folders/1NeEtFYgRbQbEyJRtSfi2h_TElcQBcLam?usp=share_link)



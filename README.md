# Healthcare Prediction System

## Overview
The Health Prediction System is a machine learning-based web application built using Streamlit. It helps users predict the likelihood of various diseases, including:
- Heart Disease
- Diabetes
- General Health Conditions

The system takes user input, processes the data, and provides a prediction along with health recommendations.

## Features
- User-friendly interface using Streamlit
- Supports prediction for multiple diseases
- Provides additional health insights and recommendations
- Input validation for reliable results
- Uses pre-trained machine learning models for predictions

## Installation

### Prerequisites
Make sure you have Python installed (version 3.7 or higher). You can install dependencies using:
```sh
pip install -r requirements.txt
```

### Running the Application
To start the Streamlit app, run the following command:
```sh
streamlit run app.py
```

## Usage
1. Open the application in your browser.
2. Select the type of disease prediction (Heart Disease, Diabetes, or General Health).
3. Enter the required health parameters.
4. Click the prediction button to get the result.
5. Review additional health suggestions and recommendations.

## Model Details
The application uses trained machine learning models for predictions. The models are trained using real-world datasets:
- **General Health Model**: Uses a combination of health indicators for assessment.
- **Heart Disease Model**: Trained using the UCI Heart Disease dataset.
- **Diabetes Model**: Trained using the Pima Indians Diabetes dataset.

## Future Enhancements
- Improve model accuracy with more training data
- Add more disease prediction categories
- Enhance UI with better visualization and user experience
- Integrate an API for real-time health monitoring

## License
This project is open-source and available under the MIT License.

## Contributors
- **Anjali Sinha**

For any questions or improvements, feel free to contribute or reach out!

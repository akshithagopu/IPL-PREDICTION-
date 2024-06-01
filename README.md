IPL Prediction Using ML
DATA SET FROM KAGGLE This web application predicts the winner of IPL matches based on user input.

Features
Predicts the winner of IPL matches based on user input regarding city, home team, away team, toss winner, toss decision, and venue.
Uses a machine learning model trained on historical IPL match data to make predictions.
Provides a simple user interface for entering input data and viewing prediction results.
Installation
Clone the repository:

Navigate to the project directory:

Install dependencies: Activate your virtual environment if you're using one for your Flask application. Install all the necessary packages using pip install or pip install -r requirements.txt. Once all packages are installed, run the command: pip freeze > requirements.txt This command will create or update the requirements.txt file with the current versions of all installed packages.

Usage
Run the Flask application:

Open a web browser and go to http://127.0.0.1:5000/ to access the web application.

Enter the required input data and submit the form to get the prediction result.

Project Structure
ipl-prediction-web-app/ │ ├── app.py # Flask application file ├── templates/ # HTML templates folder │ ├── index.html # Main page template │ └── result.html # Prediction result page template ├── static/ # Static assets folder │ └── css/ │ └── styles.css # CSS styles file ├── model.pkl # Trained machine learning model ├── requirements.txt # Python dependencies └── README.md # Project README file

License
This project is licensed under the MIT License - see the LICENSE file for details.

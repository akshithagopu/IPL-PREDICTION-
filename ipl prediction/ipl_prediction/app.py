import flask
import pickle
import pandas as pd
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=["GET", "POST"])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        city = flask.request.form['city']
        Home = flask.request.form['Home']
        Away = flask.request.form['Away']
        toss_winner = flask.request.form['toss_winner']
        toss_decision = flask.request.form['toss_decision']
        venue = flask.request.form['venue']

        if toss_winner == 'Home Team':
            toss_winner = Home
        else:
            toss_winner = Away

        input_variables = pd.DataFrame([[city, Home, Away, toss_winner, toss_decision, venue]], columns=['city', 'Home', 'Away', 'toss_winner',
        'toss_decision', 'venue'], dtype=object)

        input_variables.Home.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
                      'Rising Pune Supergiant', 'Royal Challengers Bangalore',
                      'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
                      'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
                      'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
                      np.arange(0, 14), inplace=True)
        input_variables.Away.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
                      'Rising Pune Supergiant', 'Royal Challengers Bangalore',
                      'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
                      'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
                      'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
                      np.arange(0, 14), inplace=True)
        input_variables.toss_winner.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
                             'Rising Pune Supergiant', 'Royal Challengers Bangalore',
                             'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
                             'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
                             'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
                              np.arange(0, 14), inplace=True)
        input_variables.toss_decision.replace(['bat', 'field'], [0, 1], inplace=True)
        input_variables.city.replace(['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
        'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
        'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
        'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
        'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi',
        'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah'],
        np.arange(0, 30), inplace=True)
        input_variables.venue.replace(['Rajiv Gandhi International Stadium, Uppal',
        'Maharashtra Cricket Association Stadium',
        'Saurashtra Cricket Association Stadium', 'Holkar Cricket Stadium',
        'M Chinnaswamy Stadium', 'Wankhede Stadium', 'Eden Gardens',
        'Feroz Shah Kotla',
        'Punjab Cricket Association IS Bindra Stadium, Mohali',
        'Green Park', 'Punjab Cricket Association Stadium, Mohali',
        'Sawai Mansingh Stadium', 'MA Chidambaram Stadium, Chepauk',
        'Dr DY Patil Sports Academy', 'Newlands', "St George's Park",
        'Kingsmead', 'SuperSport Park', 'Buffalo Park',
        'New Wanderers Stadium', 'De Beers Diamond Oval',
        'OUTsurance Oval', 'Brabourne Stadium',
        'Sardar Patel Stadium, Motera', 'Barabati Stadium',
        'Vidarbha Cricket Association Stadium, Jamtha',
        'Himachal Pradesh Cricket Association Stadium', 'Nehru Stadium',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Subrata Roy Sahara Stadium', 'Dr. H. A. Tyagi Cricket Stadium',
        'Jawaharlal Nehru Stadium', 'Madhavrao Scindia Cricket Stadium',
        'Hardik Pandya Cricket Stadium', 'Duleep Trophy Ground',
        'Hazare Stadium', 'Vijay Hazare Trophy Ground', 'Vinobai Panchal ground',
        'Haryana Cricket Association Stadium', 'VR Stadium',
        'Hangal ground', 'Dr. V. Gadekar Wagh Stadium',
        'Nathuvaneswarar Temple ground', 'Thiruvananthapuram',
        'NIL', 'Nil'], np.arange(0, 84), inplace=True)

    # drop variables not used in the model
    input_variables.drop(['season', 'day', 'date', 'city', 'venue'], axis=1, inplace=True)

    # save processed variables to csv file
    input_variables.to_csv('processed_variables.csv', index=False)


if __name__ == '__main__':
    # read the raw input data
    input_variables = pd.read_csv('input_variables.csv')

    # preprocess the input data
    preprocess_input_variables(input_variables)

    # read the processed input data
    processed_variables = pd.read_csv('processed_variables.csv')

    # create the target variable (0: loss, 1: win)
    processed_variables['outcome'] = processed_variables['winner'].apply(lambda x: 0 if x == 0 else 1)

    # drop unnecessary columns
    processed_variables.drop(['winner', 'margin'], axis=1, inplace=True)

    # save the preprocessed input data with the target variable to a csv file
    processed_variables.to_csv('processed_variables_with_target.csv', index=False)

    # print the shape of the preprocessed data
    print('Preprocessed data shape:', processed_variables.shape)

    # load the processed input data with the target variable
    X = processed_variables.drop('outcome', axis=1)
    y = processed_variables['outcome']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # train the classifier
    clf.fit(X_train, y_train)

    # predict the outcome for the test set
    y_pred = clf.predict(X_test)

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # print the accuracy of the model
    print('Accuracy:', accuracy)

    # save the trained model to a file
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # load the trained model from a file
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # calculate the accuracy of the loaded model
    accuracy = accuracy_score(y_test, loaded_model.predict(X_test))

    # print the accuracy of the loaded model
    print('Accuracy of loaded model:', accuracy)

    # load the actual results data
    actual_results = pd.read_csv('actual_results.csv')

    # create a column for the predicted outcomes
    actual_results['predicted_outcome'] = actual_results['match_id'].apply(lambda x: clf.predict(processed_variables[processed_variables['match_id'] == x].drop('match_id', axis=1).values[0])[0])

    # save the actual results data with the predicted outcomes to a csv file
    actual_results.to_csv('actual_results_with_predictions.csv', index=False)

    # print the confusion matrix of the model
    print('Confusion matrix:', confusion_matrix(y_test, y_pred))

    # calculate the classification report of the model
    classification_report = classification_report(y_test, y_pred)

    # print the classification report of the model
    print('Classification report:', classification_report)

    # create a dataframe to store the metrics of the model
    metrics = pd.DataFrame(columns=['metric', 'value'])

    # add the accuracy metric to the dataframe
    metrics = metrics.append({'metric': 'accuracy', 'value': accuracy}, ignore_index=True)

    # add the precision metric to the dataframe
    metrics = metrics.append({'metric': 'precision', 'value': precision_score(y_test, y_pred)}, ignore_index=True)

    # add the recall metric to the dataframe
    metrics = metrics.append({'metric': 'recall', 'value': recall_score(y_test, y_pred)}, ignore_index=True)

    # add the f1-score metric to the dataframe
    metrics = metrics.append({'metric': 'f1-score', 'value': f1_score(y_test, y_pred)}, ignore_index=True)

    # save the metrics of the model to a csv file
    metrics.to_csv('model_metrics.csv', index=False)

    # print the metrics of the model
    print('Model metrics:', metrics)# run the Flask app
    app.run(debug=True)
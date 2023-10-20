from flask import Flask, make_response, request
from flask_cors import CORS
import joblib
import pathlib
import pandas


# Load model
current_path = pathlib.Path(__file__).parent.resolve()
model = joblib.load(str(current_path)+'\\randomforest_model.joblib')


# Function to convert number into string
def to_string_evaluate(argument):
    switcher = {
        0: "Unacceptable",
        1: "Acceptable",
        2: "Good",
        3: "Very good"
    }

    return switcher.get(argument, "unknow")


# Function to convert data's string values to numeric values
def car_data_to_numeric(data):
    data_processed = data.copy()

    data_processed['buying'] = data_processed['buying'].replace({
        'low': 0,
        'med': 1,
        'high': 2,
        'vhigh': 3
    })
    data_processed['maint'] = data_processed['maint'].replace({
        'low': 0,
        'med': 1,
        'high': 2,
        'vhigh': 3
    })
    data_processed['doors'] = data_processed['doors'].replace({
        '2': 0,
        '3': 1,
        '4': 2,
        '5more': 3
    })
    data_processed['persons'] = data_processed['persons'].replace({
        '2': 0,
        '4': 1,
        'more': 2
    })
    data_processed['lug_boot'] = data_processed['lug_boot'].replace({
        'small': 0,
        'med': 1,
        'big': 2
    })
    data_processed['safety'] = data_processed['safety'].replace({
        'low': 0,
        'med': 1,
        'high': 2
    })

    return data_processed


# Create flask server
app = Flask(__name__)
# Enable CORS
CORS(app)


# Evaluate a single record from client form data and return predicted result (in string type)
@app.route('/evaluate_single', methods=['POST'])
def evaluate_single():
    # Get form data from client
    form_data = request.form.to_dict()

    # Predict the data
    predicted_result = model.predict(pandas.DataFrame(form_data, index=[0]))

    # Return predicted result to user (string)
    return to_string_evaluate(predicted_result[0])


# Evaluate multiple records from user's csv file and return predicted results through csv file 
@app.route('/evaluate_file', methods=['POST'])
def evaluate_file():
    # Get file from user
    csv_file = request.files['csv_file']

    # Read csv file
    data_origin = pandas.read_csv(csv_file)

    # Convert data's string values to numeric values
    data_processed = car_data_to_numeric(data_origin)

    # Predict car evaluation through 6 features ('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety')
    predicted_data = model.predict(
        data_processed[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']])
    
    # Convert predicted results from number to string
    predicted_results = []
    for each in predicted_data:
        predicted_results.append(to_string_evaluate(each))

    # Add predicted results to original data
    data_origin['predicted_class'] = predicted_results

    # Return download csv file to user
    response = make_response(data_origin.to_csv())
    content_disposition = 'attachment; filename=evaluated_data.csv'
    response.headers['Content-Disposition'] = content_disposition
    response.mimetype = 'text/csv'

    return response


# Run flask server
if __name__ == '__main__':
    app.run(host='localhost', port='6969')

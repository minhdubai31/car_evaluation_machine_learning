from flask import Flask, jsonify, make_response, request, send_file
from flask_cors import CORS
import joblib
import pathlib
import pandas


current_path = pathlib.Path(__file__).parent.resolve()
model = joblib.load(str(current_path)+'\\randomforest_model.joblib')


# Function to convert number into string
# Switcher is dictionary data type here
def to_string_evaluate(argument):
    switcher = {
        0: "Unacceptable",
        1: "Acceptable",
        2: "Good",
        3: "Very good"
    }

    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(argument, "nothing")

# Convert string values to numeric


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


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


@app.route('/evaluate_single', methods=['POST'])
def evaluate_single():
    form_data = request.form.to_dict()
    result = model.predict(pandas.DataFrame(form_data, index=[0]))
    print(result[0])
    return to_string_evaluate(result[0])


@app.route('/evaluate_file', methods=['POST'])
def evaluate_file():
    csv_file = request.files['csv_file']
    data_origin = pandas.read_csv(csv_file)
    data_processed = car_data_to_numeric(data_origin)

    predicted_data = model.predict(
        data_processed[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']])
    
    result = []
    for each in predicted_data:
        result.append(to_string_evaluate(each))

    data_origin['predicted_class'] = result

    response = make_response(data_origin.to_csv())
    cd = 'attachment; filename=evaluated_data.csv'
    response.headers['Content-Disposition'] = cd
    response.mimetype = 'text/csv'

    return response


if __name__ == '__main__':
    app.run(host='localhost', port='6969')

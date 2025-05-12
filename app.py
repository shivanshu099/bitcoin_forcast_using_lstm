from flask import Flask, render_template, request, jsonify
from helper import fetch_price, days_predict  # Assuming you have a helper.py file with forecast function
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    imgt = None  # Initialize imgt outside the if block
    if request.method == 'POST':
        num_of_days = request.form.get('input_data')
        if num_of_days and num_of_days.isdigit():
            img_bytes = days_predict(30, int(num_of_days)) # Increased historical data for better prediction
            if img_bytes:
                imgt = base64.b64encode(img_bytes).decode('utf-8') # Encode for embedding in HTML
            else:
                imgt = None # Handle case where prediction fails
        else:
            # Handle invalid input for number of days
            pass # You might want to display an error message to the user

    return render_template('index.html', imgt=imgt)

@app.route('/live_price')
def get_live_price():
    live_price = fetch_price()
    if live_price is not None:
        return jsonify({'price': live_price})
    else:
        return jsonify({'error': 'Could not fetch live price'}), 500


if __name__ == '__main__':
    app.run(debug=True)




    














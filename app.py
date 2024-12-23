from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load('BreastCancerPrediction_model.pkl')

# Render the HTML template
@app.route('/')
def reg():
    return render_template('reg.html')


@app.route('/home.html')
def home():
    return render_template('home.html')
#changes according to chatgpt

# Render the HTML template for the about page
@app.route('/about.html')
def about():
    return render_template('about.html')



# Render the HTML template for the contact page
@app.route('/contact.html')
def contact():
    return render_template('contact.html')

# Render the HTML template for the signup page
@app.route('/signup.html')
def signup():
    return render_template('signup.html')

# Render the HTML template for the login page
@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')


# Handle form submission
# Handle form submission for prediction
# Handle form submission for prediction
@app.route('/prediction', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        clump_thickness = float(request.form['clump_thickness'])
        unif_cell_size = float(request.form['unif_cell_size'])
        unif_cell_shape = float(request.form['unif_cell_shape'])
        marg_adhesion = float(request.form['marg_adhesion'])
        single_epith_cell_size = float(request.form['single_epith_cell_size'])
        bland_chrom = float(request.form['bland_chrom'])
        norm_nucleoli = float(request.form['norm_nucleoli'])
        mitoses = float(request.form['mitoses'])

        # Make a prediction using your trained model
        input_data = np.array([[clump_thickness, unif_cell_size, unif_cell_shape,
                                marg_adhesion, single_epith_cell_size,
                                bland_chrom, norm_nucleoli, mitoses]])
        prediction = model.predict(input_data)

        # Render the result on the HTML template
        return render_template('prediction.html',result=[prediction[0]])
    else:
        # If the request method is not POST, return an error response
        return "Method Not Allowed", 405



if __name__ == '__main__':
    app.run(debug=True)

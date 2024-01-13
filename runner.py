#import tensorflow as tf
from flask import Flask, request
import tensorflow as tf

# Load a model
model = tf.keras.models.load_model("model.keras")

# Print a summary of the model
print(model.summary())

# Create a flask application
app = Flask(__name__)

# Add a main page to the application
@app.route('/')
def index():
    return open("index.html").read()

# Add a form data handler to the application
@app.route('/', methods = ["post"])
def data():
    # Turn the received values into floats
    values = list(map(float, request.form.values()))

    # Print the values
    print(values)

    # Make a prediction on the data
    prediction = model.predict([values])

    # Send the form page and the prediction to the client
    return index() + str(prediction[0][0])

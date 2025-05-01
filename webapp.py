from flask import Flask, render_template, request, make_response
from nlp_router import predict_intent

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Create this HTML page

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_msg = request.form["msg"]
    tag, response = predict_intent(user_msg)  # Predict intent and get response

    # Create a response object and set the Content-Type header to include charset=utf-8
    response_obj = make_response(response)
    response_obj.headers['Content-Type'] = 'text/plain; charset=utf-8'
    return response_obj  # Send the response to the front-end

if __name__ == "__main__":
    app.run(debug=False)
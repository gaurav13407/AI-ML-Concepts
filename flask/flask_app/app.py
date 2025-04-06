from flask import Flask  
'''  
It creates an instance of the flask class,  
which will be your WSGI(Web Server Gateway Interface) application.  
'''  
## WSGI Application  
app = Flask(__name__)  

@app.route("/")  
def welcome():  
    return "welcome to this best flask course.This should be an amazing course"  # Corrected spelling  


@app.route("/index")
def index():
    return"Wlecome to the index page"
if __name__ == "__main__":  
    app.run(debug=True)  

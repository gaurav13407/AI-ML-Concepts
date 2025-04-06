### Bulding Url Dynamically
## Variable Rule
##jinja 2 templte Engine

## Jinja2 templte engine
'''  
{{  }} expression to print output in html
{%...%}condition, for loops
'''
from flask import Flask,render_template,request,redirect,url_for
'''  
It creates an instance of the flask class,  
which will be your WSGI(Web Server Gateway Interface) application.  
'''  
## WSGI Application  
app = Flask(__name__)  

@app.route("/")  
def welcome():  
    return "<html><H1>Wlecome to the flask Framework<H1></html>"   


@app.route("/index",methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route('/form',methods=['GET','POST'])
def form():
    if request.method=='POST':
        name=request.form['name']
        return f"hello{name}"
    return render_template('form.html')
## variable rule    
@app.route('/success/<int:score>')
def success(score):
    res=""
    if score>=50:
        res="pass"
    else:
        res="fail"
        
    return render_template('result.html',results=res)
    
    
## variable rule    
@app.route('/successre/<int:score>')
def successre(score):
    res=""
    if score>=50:
        res="pass"
    else:
        res="fail"
        
    exp={'score':score,"res":res}
    return render_template('result1.html',results=exp)

@app.route('/fail/<int:score>')
def fail(score):
   return render_template('result.html',results=score)


@app.route('/getresults',methods=['POST','GET'])
def getresult():
    pass
if __name__ == "__main__":  
    app.run(debug=True)  
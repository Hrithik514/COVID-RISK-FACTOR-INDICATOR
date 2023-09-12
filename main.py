from flask import render_template, request, redirect, url_for, jsonify, Flask
from flask_cors import cross_origin
import requests
import pymongo
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
import numpy as np
import requests
import lxml.html as lh
import pandas as pd
from bs4 import BeautifulSoup
import requests
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import cufflinks
from plotly.offline import iplot

app = Flask(__name__)
@app.route('/')
@cross_origin()
def homepage():
    return render_template("covidreport.html", msg="")

@app.route('/login', methods=['GET', 'POST'])
@cross_origin()
def login():
    collection_name = "login"
    collection = db[collection_name]
    if request.method == 'POST':
        try:
            user_name = request.form['u']
            password = request.form['p']

            authenticate = collection.find_one({"Username": user_name, "password": password})
            if authenticate is not None:
                global user_id
                user_id = authenticate['_id']
                return render_template('prediction.html')
            else:
                print("User not Found")
                return render_template('covidreport.html', msg='INVALID USER!!!')

        except RuntimeError as e:
            print(e, " : Login Authentication Failed")

@app.route('/prediction', methods=['GET', 'POST'])
@cross_origin()
def predictions():
    df = pd.read_csv("D:\VIT\Project\covidreport.csv")
    del df['Unnamed: 0']
    del df['symptoms']
    df['rtpcr'] = LabelEncoder().fit_transform(df.rtpcr)
    a = df.drop('treatment', axis=1)
    b = a.drop('severity_illness', axis=1)
    X = b.drop('len_people_infected_by_patient', axis=1)
    Y = df['treatment']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    logmodel = LogisticRegression()
    logmodel.fit(X_train, Y_train)
    a = request.form['age']
    b= request.form['p']
    c= request.form['oxi']
    d = request.form['ct']
    if (b == 'positive'):
        b = 1
    else:
        b = 0
    dat = [[a, b, c, d]]
    data = pd.DataFrame(dat, columns=['age', 'rtpcr', 'oximeter', 'ctscan'])
    prediction = logmodel.predict(data)
    a1 = df.drop('treatment', axis=1)
    b1 = a1.drop('severity_illness', axis=1)
    X1 = b1.drop('len_people_infected_by_patient', axis=1)
    Y1 = df['severity_illness']
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=0)
    gnb = GaussianNB().fit(X1_train, Y1_train)
    gnb_predictions = gnb.predict(data)

    return render_template('prediction.html' , prediction=prediction[0],gnb=gnb_predictions[0])

@app.route('/world', methods=['GET', 'POST'])
@cross_origin()
def worlds():
    a=datetime.now().strftime('%d-%m-%Y %I:%M %p')
    cufflinks.go_offline()
    wd_url = "https://www.worldometers.info/coronavirus/"
    req = requests.get(wd_url)
    data = req.text
    soup = BeautifulSoup(data, 'html.parser')
    table_body = soup.find('tbody')
    table_rows = table_body.find_all('tr')
    country = []
    totalcases = []
    newcases = []
    totaldeaths = []
    newdeaths = []
    totalrecovered = []

    for tr in table_rows:
        td = tr.find_all('td')
        country.append(td[1].text)
        totalcases.append(td[2].text.strip())
        newcases.append(td[3].text.strip())
        totaldeaths.append(td[4].text.strip())
        newdeaths.append(td[5].text.strip())
        totalrecovered.append(td[6].text)
    df1 = pd.DataFrame({'Country': country, 'TotalCases': totalcases, 'NewCases': newcases,
                        'TotalDeaths': totaldeaths, 'NewDeaths': newdeaths, 'TotalRecovered': totalrecovered})
    df = df1.loc[8:]
    m = []
    for i in range(224):
        n = (df.iloc[i]).to_dict()
        m.append(n)
    df_world = pd.DataFrame(df1.loc[7])
    df_world.drop(["Country"], inplace=True)
    c = df_world.head().to_dict()

    def null_values(data):
        data.fillna(0, inplace=True)
        return data

    df = null_values(df)

    def preprocess(data):
        for cols in data:
            data[cols] = data[cols].map(lambda x: str(x).replace(',', ''))
            data[cols] = data[cols].map(lambda x: str(x).replace('+', ''))
            data[cols] = data[cols].map(lambda x: str(x).replace('\n', ''))

    preprocess(df_world)
    preprocess(df)
    d = df_world.head().to_dict()

    return render_template('world.html', a=a, b=m, c=c, d=d)
@app.route('/advisory', methods=['GET', 'POST'])
@cross_origin()
def advisory():
    return render_template('advisory.html',msg="")

@app.route('/beds', methods=['GET', 'POST'])
@cross_origin()
def hospitals():
    a = datetime.now().strftime('%d-%m-%Y %I:%M %p')
    data1 = pd.read_csv("D:/VIT/Project/beds1.csv")
    del data1['Unnamed: 0']
    u = []
    for i in range(394):
        v = (data1.iloc[i]).to_dict()
        u.append(v)
    return render_template('beds.html', a1=a, beds=u)

@app.route('/reqs', methods=['GET', 'POST'])
@cross_origin()
def req():
    return render_template('request.html',msg="")

if __name__ == '__main__':
    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
    DB_NAME = "Covidreport"
    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)
    db = client[DB_NAME]
    app.run(debug=True)

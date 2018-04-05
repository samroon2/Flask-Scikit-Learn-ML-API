from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import os


app = Flask(__name__)

#route our app to domain/predictiris , accept POST requests
@app.route('/predictiris', methods=['POST'])

def irisapi():
	'''our main function that handles our requests and delivers classified data in json format'''

	#Recieve our json data from our client, create a dataframe.
	try:
		postjson = request.json
		clientdata = pd.DataFrame(postjson)
	except Exception as e:
		raise e

	#Evaluate whether we have data, if we don't return bad_request	
	if clientdata.empty:
		return(bad_request())
	
	#load our iris clf model
	pklclf = 'irisclassifier.pkl'
	clf = joblib.load(f'./models/{pklclf}')

	#classify our clients data
	y_pred = clf.predict(clientdata)

	#Decode our encoded data lables, so the client recieves lables not encoded values
	fnames = {0:'Versicolor',1:'Virginica',2:'Setosa'}

	#list comprehension, applying our function to get the labels and placing in our dataframe			
	clientdata['PredictedIris'] = [fnames.get(x, x) for x in y_pred]

	#return json data to our client, give status code 200 - OK response contains a payload
	apiresponse = jsonify(predictions=clientdata.to_json(orient="records"))
	apiresponse.status_code = 200

	return (apiresponse)

if __name__ == "__main__":
	app.run()
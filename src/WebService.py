from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import  joblib
import json
import sys
from Lib import  *
from Commons import  *
import numba

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # request data

        model = joblib.load('C:/Users/marco.fumagalli/LightFM/Cached/lightfmobj.pkl')
        item_lookup = joblib.load('C:/Users/marco.fumagalli/LightFM/Cached/item_lookup.pkl')

        request_json = request.get_json(force=True)
        jsondata = json.dumps(request_json)

        try:
            jsondata = json.loads(jsondata)
        except Exception as e:
            logger_info = SetupLogger("C:/Users/marco.fumagalli/LightFM/Log/logs.log", "INFO")
            logger_info.info('Error in loading json: {}.'.format(e))
            return -1

        user = jsondata['userid']
        rec = model.rec_items(user, model.artid, model.userid, 5, item_lookup)

        rec = jsonify(rec)

        return rec

if __name__ == '__main__':
    sys.path.append("C:/Users/marco.fumagalli/LightFM")

    # start api
    #app.run(host='0.0.0.0', port=9998, debug=True)
    app.run(debug=True)
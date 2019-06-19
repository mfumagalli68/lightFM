from src.Lib import *
import os
from src.model.Model import TuneModel,LightFMModel
import datetime

os.chdir('C:\\Users\\marco.fumagalli\\LightFM')

#ImportAll()


items = pd.read_pickle("src/data/item_data.pkl")
article_to_exclude = pd.read_pickle("src/data/article_to_exclude.pkl")
useritem = pd.read_pickle("src/data/UserItem.pkl")
item_lookup = pd.read_pickle("src/data/item_lookup.pkl")
usercustomerdata = pd.read_pickle("src/data/usercustomer.pkl")
userdata = pd.read_pickle("src/data/UserData.pkl")

if 'hyperparams.pkl' in os.listdir('models'):
        creattime = datetime.datetime.fromtimestamp(os.path.getmtime('models/hyperparams.pkl'))
        timediff = datetime.datetime.today() - creattime
        if timediff.days > 20:
            tuneModel = TuneModel()
            tuneModel.hyperparametertuning()
        else:
            hyperparams = joblib.load('models/hyperparams.pkl')

else:
    tuneModel = TuneModel((10,0.01,25,0.00001),useritem)
    tuneModel.hyperparametertuning()
    hyperparams = joblib.load('models/hyperparams.pkl')

print ("beginning model...")


mod = LightFMModel(hyperparams['no_components'],
                   hyperparams['learning_rate'],
                   hyperparams['epochs'],
                   useritem, items, usercustomerdata,hyperparams['alpha'])

mod = joblib.load('C:\\Users\\marco.fumagalli\\LightFM\\Cached\\mod_obj.pkl')
customers_arr = np.array(mod.userid)

auc , pr = mod.compute_auc_prec(5)
print ("auc {}".format(auc) )
print ("precision at 5 {}".format(pr) )

# Prova recommending items for a particular customers

rec = mod.rec_items('8928555001',mod.artid,customers_arr,10,item_lookup)
df_filt = useritem[useritem['USERID']=='8928555001']
df_filt = pd.merge(df_filt,item_lookup,on='ARTID')
print(rec.values())
print ("actual items seen {}".format(df_filt.ARTDSC.unique()))

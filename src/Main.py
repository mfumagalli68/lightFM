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

#mod = joblib.load('Cached/model.pkl')

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

print("done")

#rec = mod.rec_items('0201135001',mod.artid,customers_arr,5,item_lookup)

#whatbought = useritem.loc[useritem.USERID=='0201135001']
#whatbought = pd.merge(whatbought,item_lookup,how="inner",on="ARTID")
#print (rec)
#print(whatbought)

mod = joblib.load('C:\\Users\\marco.fumagalli\\LightFM\\Cached\\mod_obj.pkl')
customers_arr = np.array(mod.userid)

auc , pr = mod.compute_auc_prec(5)
#pers_score = mod.personalization_score(5)
print ("auc {}".format(auc) )
print ("precision at 5 {}".format(pr) )
#print ("personal score {}".format(pers_score) )
rec = mod.rec_items('8928555001',mod.artid,customers_arr,10,item_lookup)
df_filt = useritem[useritem['USERID']=='8928555001']
df_filt = pd.merge(df_filt,item_lookup,on='ARTID')
print(rec.values())
print ("actual items seen {}".format(df_filt.ARTDSC.unique()))
#print (datetime.datetime.now().strftime("%Y-%M-%d %H-%M-%S"))

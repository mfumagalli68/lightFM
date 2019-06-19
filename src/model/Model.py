from lightfm import LightFM
from src.ConnectionDB import *
import numpy as np
from src.Commons import make_train,SetupLogger
from sklearn import metrics
import pandas as pd
import datetime
import joblib
from src.CreateMatrix import CreateItemMatrix,CreateUserItemMatrix,CreateUserMatrix
import os
from lightfm import LightFM
from lightfm import evaluation
from skopt import forest_minimize
import numba

class LightFMModel():
    """
    Class LightFMModel:
    """
    def __init__(self,latent_factor,learning_rate,n_epochs,useritem,itemmatrix,usercustomerdata,alpha):

        self.item_feature = CreateItemMatrix(itemmatrix)
        self.userid, self.artid, self.matrix = CreateUserItemMatrix(useritem)
        #self.user_feature=CreateUserMatrix(usermatrix,self.userid)
        self.latent_factor = latent_factor
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.train_test_user = self.make_train_test()
        self.order_data = usercustomerdata
        self.user_alpha = alpha
        self.item_alpha = alpha
        self.model=self.fit_model()



    def make_train_test(self):
        return make_train(self.matrix)


    def fit_model(self):
        """
        Method to fit model. If model was already fitted just load it with joblib and
        execute fit_partial
        :return:
        """

        file_list = os.listdir('./models')
        #if 'model.pkl' in file_list:
        #    logger_info = SetupLogger("Log/log.txt", "INFO")
        #    logger_info.info("Loading model and beginning refitting")
        #    joblib.load('Cached/model.pkl')

        model = LightFM(no_components=self.latent_factor, learning_rate=self.learning_rate, loss='warp',
                        user_alpha=self.user_alpha,
                        item_alpha=self.item_alpha
                        )
        model.fit(self.train_test_user[0],
                            epochs=self.n_epochs,
                            item_features=self.item_feature,
                            #user_features=self.user_feature,
                            num_threads=4,
                            verbose=True)

        joblib.dump(model,'models/model.pkl')
        logger_info = SetupLogger("Log/log.txt", "INFO")
        logger_info.info("Model fitted.")

        return model

    def make_prediction(self,user_id):
        """
        Method to make predictions.
        :param user_id: Userid or customerid? (also as a list)
        :return:
        """
        #TODO define if user_id will be CUSTOMERID or USERID
        n_items = self.matrix.shape[1]
        pid_array = np.arange(n_items, dtype=np.int32)
        uid_array = np.empty(n_items, dtype=np.int64)

        uid_array.fill(np.asscalar(np.array(user_id)))
        predictions = self.model.predict(
            uid_array,
            pid_array,
            item_features=self.item_feature,
            #user_features=self.user_feature,
            num_threads=4)

        return predictions


    def auc_score(self,predictions, target):
        """

        :param predictions: predictions from make_prediciton method
        :param target: target array
        :return: auc
        """


        fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
        return metrics.auc(fpr, tpr)


    def precision_at_k(self,predictions,target,k):

        index_actual = target.nonzero()
        shape_index=index_actual[0].shape[0]
        if shape_index <k:
            k=shape_index
        top_k_rec = predictions[index_actual][:k]

        return sum([1 if i > 1 else 0 for i in top_k_rec])/k


    def compute_auc_prec(self,k):
        """
        Compute auc for test set.
        :return:
        """
        store_auc = []
        store_precision=[]
        #popularity_auc = []
        #pop_items = np.array(self.train_test_user[1].sum(axis=0)).reshape(-1)

        for u in self.train_test_user[2]:

            pred = self.make_prediction(u).reshape(-1)
            #pred = [index for index, value in sorted(enumerate(pred), reverse=True, key=lambda x: x[1])][:20]
            actual = self.train_test_user[1][u,:].toarray().reshape(-1)
            actual[np.where(actual>1)]=1
            store_precision.append(self.precision_at_k(pred,actual,k))
            try:
                store_auc.append(self.auc_score(pred, actual))
            except:
                print("error at users {}".format(u))

        return (np.mean(store_auc), np.mean(store_precision))


    def rec_items(self,user_id,item_list,user_id_list,num_items,item_lookup):

        # Load items

        usercustomer = pd.read_pickle("C:\\Users\\marco.fumagalli\\LightFM\\src\\data\\usercustomer.pkl")
        usercustomer = usercustomer.loc[usercustomer.USERID == user_id]


        reclist=[]
        final_dict={}
        cust_ind = np.where(user_id == np.array(user_id_list))[0]
        print(cust_ind)

        logger_info = SetupLogger("Log/logs.log", "INFO")
        logger_info.info('Processing customer: {}'.format(cust_ind))

        pred = self.make_prediction(cust_ind)
        product_idx = np.argsort(pred)[::-1][:num_items]

        for indx in product_idx:
            code=item_list[indx]
            if code not in usercustomer.ARTID.tolist():
                try:
                    reclist.append([code,item_lookup.ARTDSC.loc[item_lookup.ARTID==code].iloc[0]])
                except Exception as e:
                    print (e)
                    logger_info = SetupLogger("Log/logs.log","INFO")
                    logger_info.info('Got an exception. Can t find an item description for '
                                      'article code {}'.format(e))

        codes = [item[0] for item in reclist]
        descriptions = [item[1] for item in reclist]

        #final_dict[user_id]=(codes,descriptions)
        final_dict[user_id] = codes

        #joblib.dump(final_dict,'Recommendation/recommendation.pkl')
        return final_dict

    def personalization_score(self,num_items):
        '''
        :param recommended_matrix: matrix with user on rows and recommendation on columns
        :return:
        '''
        recommended = []
        for u in self.train_test_user[2]:
            pred = self.make_prediction(u)
            product = np.argsort(pred)[::-1][:num_items]
            recommended.append(product)

        recommended = np.array(recommended)
        cosine_similarity = metrics.pairwise.cosine_similarity(recommended)
        pers = 1-np.triu(cosine_similarity,1).mean()
        logger_info = SetupLogger("Log/logs.log", "INFO")
        logger_info.info('Personalization score computed')
        return pers





class TuneModel(LightFMModel):

    def __init__(self,params,useritem):
        self.params = params
        self.userid, self.artid, self.matrix = CreateUserItemMatrix(useritem)
        self.train_test_user = self.make_train_test()
        #self.objective = self.objective(self.params)


    def objective(self, params):

        epochs, learning_rate, no_components, alpha = params

        user_alpha = alpha
        item_alpha = alpha
        model = LightFM(loss='warp',
                            random_state=2016,
                            learning_rate=learning_rate,
                            no_components=no_components,
                            user_alpha=user_alpha,
                            item_alpha=item_alpha)

        model.fit(self.train_test_user[0], epochs=epochs,
                      num_threads=4, verbose=True)

        patks = evaluation.precision_at_k(model, self.train_test_user[1],
                                              train_interactions=None,
                                              k=5, num_threads=4)
        print("running hyperparmeter.." + datetime.datetime.now().strftime("%Y-%M-%d %H:%m"))
        mapatk = np.mean(patks)
            # Make negative because we want to _minimize_ objective
        out = -mapatk
            # Handle some weird numerical shit going on
        if np.abs(out + 1) < 0.01 or out < -1.0:
            return 0.0
        else:
            return out

    @numba.jit
    def hyperparametertuning(self):

        space = [(50, 100),  # epochs
                 (10 ** -4, 0.2, 'log-uniform'),  # learning_rate
                 (5, 35),  # no_components
                 (10 ** -6, 10 ** -1, 'log-uniform'),  # alpha
                 ]

        res_fm = forest_minimize(self.objective, space, n_calls=20,
                                 random_state=0,
                                 verbose=True)

        paramsdict = {}
        params = ['epochs', 'learning_rate', 'no_components', 'alpha']
        for (p, x_) in zip(params, res_fm.x):
            paramsdict[p] = x_

        joblib.dump(paramsdict, 'models/hyperparams.pkl')







# if __name__=='__main__':
#     import joblib
#
#     data = joblib.load('cleaned_retail.pkl')
#
#     item_lookup = data[
#         ['StockCode', 'Description']].drop_duplicates()  # Only get unique item/description pairs
#     item_lookup['StockCode'] = item_lookup.StockCode.astype(str)  # Encode as strings for future lookup ease
#     #
#     data['CustomerID'] = data.CustomerID.astype(int)  # Convert to int for customer ID
#     cleaned_retail = data[['StockCode', 'Quantity', 'CustomerID']]  # Get rid of unnecessary info
#     grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index()  # Group together
#     grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1  # Replace a sum of zero purchases with a one to
#     # # indicate purchased
#     grouped_purchased = grouped_cleaned.query('Quantity > 0')  # Only get customers where purchase totals were positive
#     #
#     customers = list(np.sort(grouped_purchased.CustomerID.unique()))  # Get our unique customers
#     products = list(grouped_purchased.StockCode.unique())  # Get our unique products that were purchased
#     quantity = list(grouped_purchased.Quantity)  # All of our purchases
#     #
#     rows = grouped_purchased.CustomerID.astype('category', categories=customers).cat.codes
#     # # Get the associated row indices
#     cols = grouped_purchased.StockCode.astype('category', categories=products).cat.codes
#
#     customers_arr = np.array(customers)  # Array of customer IDs from the ratings matrix
#     products_arr = np.array(products)
#
#     matrix = joblib.load('purchases_sparse.pkl')
#
#
#     mod = LightFMModel(matrix, 20, 0.01, 100)
#     mod.compute_auc()
#     mod.personalization_score(10)
#     mod.rec_items(12346,products_arr,customers_arr,10,item_lookup)

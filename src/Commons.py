from src.Lib import *
import random
import logging
from sklearn.model_selection import train_test_split


def ValidationDays(data,split):

    len_data = data.shape[0]
    split_data = np.floor(len_data*split)

    train = data[:split_data]
    train['SET']='TRAIN'

    test = data[:split_data+1:len_data]
    train['TEST'] = 'VALIDATION'

    train = train.reset_index()
    test = test.reset_index()

    #TODO save info about train and test

    return train, test


# def make_train(ratings, pct_test=0.01):
#     """
#     This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
#     user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings,
#     while the training set replaces the specified percentage of them with a zero in the original ratings matrix.
#
#     :param ratings: The altered version of the original data with a certain percentage of the user-item pairs
#     that originally had interaction set back to zero.
#     :param pct_test: The percentage of user-item interactions where an interaction took place that you want to mask in the
#     training set for later comparison to the test set, which contains all of the original rating
#     :return: A list with
#
#     training_set: Altered version of the original data with a certain percentage of the user-item pairs
#                   that originally had interaction set back to zero.
#     test_set -    A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order
#                   compares with the actual interactions.
#     user_inds -   From the randomly selected user-item indices, which user rows were altered in the training data.
#                   This will be necessary later when evaluating the performance via AUC.
#     """
#
#
#
#     # Create random index
#
#     #joblib.load('purchases_sparse.pkl')
#     test_set = ratings.copy()  # Make a copy of the original set to be the test set.
#     test_set[test_set != 0] = 1  # Store the test set as a binary preference matrix
#     training_set = ratings.copy()  # Make a copy of the original data we can alter as our training set.
#     nonzero_inds = training_set.nonzero()  # Find the indices in the ratings data where an interaction exists
#     nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))  # Zip these pairs together of user,item index into list
#     random.seed(0)  # Set the random seed to zero for reproducibility
#     num_samples = int(
#         np.ceil(pct_test * len(nonzero_pairs)))  # Round the number of samples needed to the nearest integer
#     samples = random.sample(nonzero_pairs, num_samples)  # Sample a random number of user-item pairs without replacement
#     user_inds = [index[0] for index in samples]  # Get the user row indices
#     item_inds = [index[1] for index in samples]  # Get the item column indices
#     training_set[user_inds, item_inds] = 0  # Assign all of the randomly chosen user-item pairs to zero
#     training_set.eliminate_zeros()  # Get rid of zeros in sparse array storage after update to save space
#
#     return training_set, test_set, list(set(user_inds))  # Output the unique list of user rows that were altered

# Train and test on not sparse matrix... can'
#https://jessesw.com/Rec-System/


def make_train_dense(data,pct_test=0.2):
    useritem = pd.read_pickle("Cached/UserItem.pkl")
    train, test = train_test_split(useritem,test_size=pct_test,stratify="USERID")

def make_train(ratings,pct_test=0.1):

    user_index = range(ratings.shape[0])
    train = ratings.copy()
    test = sparse.csr_matrix(train.shape)
    for user in user_index:

        num_samples = int(ratings.getrow(user).indices.shape[0]*pct_test)
        #test_ratings = np.random.choice(ratings.getrow(user).indices,
        #                            size=num_samples,
        #                            replace=False)
        test_ratings = ratings.getrow(user).indices[-num_samples:]

        train[user, test_ratings] = 0
        test[user, test_ratings] = ratings[user, test_ratings]


    train.eliminate_zeros()
    nonzero_inds_test = test.nonzero()
    user_inds = list(set(nonzero_inds_test[0]))
    return train, test,user_inds

def SetupLogger(log_file, level):

    """
    :param log_file: log_file with extension
    :param level: level for logging (logging.error,logging.info...)
    :return: logger
    """

    # pdb.set_trace()
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    if (logger.hasHandlers()):
        logger.handlers.clear()
        logger.addHandler(file_handler)
    else:
        logger.addHandler(file_handler)

    return logger

def kpi_purchase(self,whichDB):

    """
    :param self:
    :return:
    """

    data = LoadUserCustomerData(whichDB,3)


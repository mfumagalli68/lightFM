from src.Lib import *


def CreateItemMatrix(matrix):
    '''
    :param matrix: Item matrix with groupdsc and quickvalue columns
    :return: a sparse matrix csr
    '''

    # Need to clean data before build matrix

    # Probabilmente è meglio legare GROUPDSC1 con QUICKDISC_1

    cols_q = ['QUICKVALUE_1','QUICKVALUE_2','QUICKVALUE_3','QUICKVALUE_4']

    cols_g = ['GROUPDSC_1','GROUPDSC_2',
            'GROUPDSC_3','GROUPDSC_4']

    matrix = matrix.apply(lambda x: x.str.upper())
    for i,j in zip(cols_g,cols_q):
        matrix[i+'-'+j] = matrix[i]+'-'+matrix[j]
        matrix=matrix.drop(columns=[i,j])


    matrix = matrix.melt(id_vars='ARTID',value_name='INFO')
    matrix = matrix.drop(columns=['variable'])
    matrix['QUANTITY']=1

    # drop na
    matrix = matrix.dropna(axis=0)
    matrix = matrix.drop_duplicates()

    artid = list(np.sort(matrix.ARTID.unique()))  # Get our unique customers
    info = list(matrix.INFO.unique())  # Get our unique products that were purchased
    quantity = list(matrix.QUANTITY)

    #rows = matrix.ARTID.astype('category', categories=artid).cat.codes
    rows = pd.Categorical(matrix.ARTID, categories=artid).codes

    # Get the associated row indices
    #cols = matrix.INFO.astype('category', categories=info).cat.codes
    cols = pd.Categorical(matrix.INFO, categories=info).codes

    # Get the associated column indices
    item_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(artid), len(info)))

    # Need to save everything i need to


    return item_sparse

def CreateUserItemMatrix(matrix):

    """
    :param matrix: UserItem Matrix
    :return: tuple with: a sparse matrix (csr). rows are userid, cols are artid. Return also userid and artid list
    """
    to_exclude = pd.read_pickle("src/data/article_to_exclude.pkl")
    to_exclude = to_exclude.drop(columns=['CATEGORYDSC'])

    matrix = matrix.groupby(['DATE','USERID','ARTID'],as_index=False).agg({'TOTALCLICK':'sum'}).sort_values(by="DATE")
    matrix['TOTALCLICK']=1

    matrix = matrix.reset_index(drop=True)

    df_filt = matrix.merge(to_exclude.drop_duplicates(), on=['ARTID'],
                       how='left', indicator=True)
    df_filt = df_filt[df_filt['_merge'] == 'left_only']
    df_filt = df_filt.drop(columns=['_merge'])

    userid = list(np.sort(df_filt.USERID.unique()))  # Get our unique customers
    artid = list(df_filt.ARTID.unique())  # Get our unique products that were purchased
    click = list(df_filt.TOTALCLICK)

    rows = pd.Categorical(df_filt.USERID, categories=userid).codes

    # Get the associated row indices
    cols = pd.Categorical(df_filt.ARTID, categories=artid).codes

    # Get the associated column indices
    item_sparse = sparse.csr_matrix((click, (rows, cols)), shape=(len(userid), len(artid)))

    return userid,artid,item_sparse


def CreateUserMatrix(matrix,userid):

    """
    :param matrix: UserItem Matrix
    :return: tuple with: a sparse matrix (csr). rows are userid, cols are artid. Return also userid and artid list
    """
    #df = pd.read_pickle("./article_to_exclude.pkl")

    matrix = matrix.drop(columns=['CATEGORYID'])

    matrix = matrix.loc[matrix['USERID'].isin(userid)]

    userid_mat = list(np.sort(matrix.USERID.unique()))  # Get our unique customers
    filtro_scelta = list(matrix.FILTRO_SCELTA.unique())  # Get our unique products that were purchased
    click = list(matrix.TOTALCLICK)

    rows = pd.Categorical(matrix.USERID, categories=userid_mat).codes

    # Get the associated row indices
    cols = pd.Categorical(matrix.FILTRO_SCELTA, categories=filtro_scelta).codes

    # Get the associated column indices
    user_sparse = sparse.csr_matrix((click, (rows, cols)), shape=(len(userid), len(filtro_scelta)))

    return user_sparse


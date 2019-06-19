from src.ConnectionDB import *
from src.Lib import *


def LoadDataItem(whichDB):

    con_obj = DBConnection(whichDB)
    con = con_obj._open_connection_vertica()

    try:
        con.cursor().execute("""SELECT ESP_DQUICKINFO.ARTID,ESP_DQUICKINFO.CATEGORYDSC,
                    ESP_DQUICKINFO.QUICKDSC_1,
                    ESP_DQUICKINFO.QUICKVALUE_1, 
                    ESP_DQUICKINFO.QUICKDSC_2,ESP_DQUICKINFO.QUICKVALUE_2,
                    ESP_DQUICKINFO.QUICKDSC_3,ESP_DQUICKINFO.QUICKVALUE_3,
                    ESP_DQUICKINFO.QUICKDSC_4,ESP_DQUICKINFO.QUICKVALUE_4,
                    ESP_DQUICKINFO.ARTDSC
                    FROM ESPDDS.ESP_DQUICKINFO
                    WHERE ESP_DQUICKINFO.GROUPDSC_1 IS NOT NULL AND ESP_DQUICKINFO.GROUPDSC_2 IS NOT NULL 
                    AND ESP_DQUICKINFO.GROUPDSC_3 IS NOT NULL AND ESP_DQUICKINFO.COMPANYID=1
                    """)


        data = con.cursor().fetchall()
        df = pd.DataFrame(data)
        df.columns=['ARTID','CATEGORYDSC','GROUPDSC_1',
                               'QUICKVALUE_1', 'GROUPDSC_2', 'QUICKVALUE_2', 'GROUPDSC_3',
                               'QUICKVALUE_3','GROUPDSC_4','QUICKVALUE_4','ARTDSC']

        item_lookup = df[['ARTID','ARTDSC']]
        item_lookup.to_pickle('src/data/item_lookup.pkl')
        df = df.drop(columns=['ARTDSC'])

    except Exception as e:

        logger_info = SetupLogger("Log/logs.log", "ERROR")
        logger_info.info('Got an exception. Error retrieving data {}'.format(e))

    try:
        #Product for which we don't have any quick info at DB
        con.cursor().execute(""" SELECT DISTINCT ESP_DQUICKINFO.ARTID,
                    ESP_DQUICKINFO.CATEGORYDSC
                    FROM ESPDDS.ESP_DQUICKINFO
                 
                    WHERE ESP_DQUICKINFO.GROUPDSC_1 IS NULL AND ESP_DQUICKINFO.GROUPDSC_2 IS NULL
                    AND ESP_DQUICKINFO.GROUPDSC_3 IS NULL AND ESP_DQUICKINFO.COMPANYID=1""")

        article_to_exclude = con.cursor().fetchall()
        article_to_exclude = pd.DataFrame(article_to_exclude)
        article_to_exclude.columns=['ARTID','CATEGORYDSC']

        article_to_exclude.to_pickle("Cached/article_to_exclude.pkl")

        df = df[~df['ARTID'].isin(article_to_exclude)]
        df.to_pickle("src/data/item_data.pkl")

    except Exception as e:
        logger_info = SetupLogger("Log/logs.log", "ERROR")
        logger_info.info('Got an exception. Error retrieving data {}'.format(e))

    con.close()


def LoadDataUserItem(whichDB):

    """
    Load data User Item
    :return:  Cached dataset in Cached folder
    """

    con_obj = DBConnection(whichDB)
    con = con_obj._open_connection_vertica()


    try:
        con.cursor().execute("""
             SELECT ux.USERID,ux.AZIONE AS ARTID,TOTALCLICK,ux.DATE
                    FROM GoogleAnalytics.UX_RISULTATI_RICERCA as ux
                    WHERE COMPANYID=1
                    ORDER BY ux.DATE ASC
                    """)
        data = con.cursor().fetchall()
        df = pd.DataFrame(data)
        df.columns = ["USERID", "ARTID", "TOTALCLICK",'DATE']
        df.to_pickle("src/data/UserItem.pkl")

    except Exception as e:

        logger_info = SetupLogger("Log/logs.log", "ERROR")
        logger_info.info('Got an exception. Error retrieving data {}'.format(e))

    con.close()


def LoadUserCustomerData(whichDB,daysbehind):
    """
    Load UserCustomer association. Filter for customer for which we have a recommendation
    This dataset will be cached daily.
    When web service will call to get recommended items for a users i will check on this table if any
    of the product recommended was already ordered yesterday. If so, delete it.

    :return:
    Data cached in Cached Folder
    """

    # TODO Find in Oracle UserId and customer


    con_obj = DBConnection(whichDB)
    con = con_obj._open_connection_oracle()


    try:
        query = """
                   SELECT ESPC.CUSTOMERID, ESPCC.USERID, ESPC.ARTID, ESPC.ORDERNUM
                   FROM ESPDDS.ESP_FCORDERS ESPC
                   LEFT JOIN ESPDDS.ESP_DCUSTOMER_CONTACTS ESPCC
                   ON ESPCC.CUSTOMERID=ESPC.CUSTOMERID
                   WHERE ESPC.COMPANYID=1 AND ESPC.INSDATE> current_date-(%s)
                   
                 """ %daysbehind

        data = pd.read_sql(query, con=con)
        data.to_pickle('src/data/usercustomer.pkl')

    except Exception as e:

        logger_info = SetupLogger("Log/logs.log", "ERROR")
        logger_info.info('Got an exception. Error retrieving data {}'.format(e))



    con.close()


def LoadUserData(whichDB):

    """
    Load data User Item
    :return:  Cached dataset in Cached folder
    """

    con_obj = DBConnection(whichDB)

    con = con_obj._open_connection_vertica()

    try:
        con.cursor().execute("""
            SELECT X.USERID,X.CATEGORYID,X.FILTRO_SCELTA,COUNT(X.TOTALCLICK) AS TOTCLICK
            FROM(
            SELECT CONCAT(CONCAT(UPPER(TIP.FILTRO),'-'),UPPER(TIP.SCELTA)) AS FILTRO_SCELTA,TIP.CATEGORYID,TIP.USERID,TIP.TOTALCLICK
            FROM GoogleAnalytics.UX_FILTRI_ESTESI_CLICK_TIPOLOGIA  AS TIP
            where TIP.SCELTA IS NOT NULL AND TIP.COMPANYID=1
            ) X
            group by X.USERID,X.FILTRO_SCELTA,X.CATEGORYID

                    """)

        data = con.cursor().fetchall()
        df = pd.DataFrame(data)
        df.columns = ["USERID", "CATEGORYID", "FILTRO_SCELTA",'TOTALCLICK']
        df.to_pickle("src/data/UserData.pkl")

    except Exception as e:

        logger_info = SetupLogger("Log/logs.log", "ERROR")
        logger_info.info('Got an exception. Error retrieving data {}'.format(e))


    con.close()


def ImportAll():

    LoadDataItem("Vertica")
    LoadDataUserItem("Vertica")
    LoadUserCustomerData("Oracle",30)


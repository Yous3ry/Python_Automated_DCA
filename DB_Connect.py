import pandas as pd
import json
from sqlalchemy.engine import URL
from sqlalchemy.engine import create_engine


def conn_DB(login):
    """
    Connects to production Database
    Input: login credentials dictionary
    :return: Connection engine
    """
    # Define Connection
    connection_string = '''DRIVER=SQL SERVER;SERVER={}; 
                        DATABASE={};
                        UID={};PWD={};
                        Trusted_Connection=no;'''.format(login["SERVER"], login["DB"], login["UID"], login["PWD"])
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
    engine = create_engine(connection_url)
    return engine


def read_prod_data(connection, table, target_well):
    """
    Reads data from production database
    :input: database connection engine, table name in DB, target well name
    :return: Production & injection df and Well locations & types df
    """
    # read SQL as dataframe
    df = pd.read_sql('''SELECT * FROM {}
    WHERE ITEM_NAME LIKE '{}';'''.format(table, target_well), connection)
    return df


# reading log in credentials from text file
with open("D:\\Logins.txt") as file:
    login_data = json.loads(file.read())  # Data stored as a JSON format (Dictionary)
print("Login Credentials", login_data)

# create connection
conn = conn_DB(login_data)

# Define Database table to read data from
prod_table = "VT_TOTALS_DAY_en_US"
well = "BERENICE-01"

# returns desired well production dataframe
prod_df = read_prod_data(conn, prod_table, well)
print(prod_df)

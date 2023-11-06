import pyodbc
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

# get credentials from the Azure ML workspace service principal
credential = ManagedIdentityCredential()
# access the KeyVault in another resource group
KVUri = "https://bdf-dataeng-kv.vault.azure.net"
kVClient = SecretClient(vault_url=KVUri, credential=credential)

# create datalake connection
def get_dl_conn():
    con = pyodbc.connect(
        driver="{ODBC Driver 17 for SQL Server}",
        server=kVClient.get_secret("datalake-sqlpool-server-prd").value,
        port=1433,
        database=kVClient.get_secret("datalake-sqlpool-dbname-prd").value,
        uid=kVClient.get_secret("datalake-sqlpool-user-prd").value,
        pwd=kVClient.get_secret("datalake-sqlpool-psw-prd").value,
    )
    print("Connect to Database \U00002705")
    return con

from duckdb import DuckDBPyConnection

from dbt.adapters.duckdb.plugins import BasePlugin
from dbt.adapters.duckdb.utils import TargetConfig

from rumah_transformers import mask_name, mask_phone, clean_facility, clean_agency_company


# The python module that you create must have a class named "Plugin"
# which extends the `dbt.adapters.duckdb.plugins.BasePlugin` class.
class Plugin(BasePlugin):

    def configure_connection(self, conn: DuckDBPyConnection):
        conn.create_function("MASK_NAME", mask_name)
        conn.create_function("MASK_PHONE", mask_phone)
        conn.create_function("CLEAN_FACILITY", clean_facility)
        conn.create_function("CLEAN_AGENCY_COMPANY", clean_agency_company)

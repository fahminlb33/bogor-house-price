import pandas as pd

def model(dbt, session):
	# load spatial price dataset
	df_spatial = dbt.ref("marts_spatial_price").df()

	return df_spatial.corr(numeric_only=True)

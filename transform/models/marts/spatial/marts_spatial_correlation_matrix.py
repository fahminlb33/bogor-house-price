import numpy as np

def model(dbt, session):
	# load spatial price dataset
	df_spatial = dbt.ref("marts_spatial_price").df()
	df_spatial["price"] = np.log(df_spatial["price"])

	return df_spatial.corr(numeric_only=True)

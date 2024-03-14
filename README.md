# House Price Analysis in Bogor

This is a complete end-to-end project for house price analysis and prediction at Kabupaten Bogor and Kota Bogor. What's included:

- web scraping using scrapy
- ETL pipeline using DuckDB and dbt
- exploratory data analysis using pandas
- predictive modelling
- hyperparameter search using optuna

The runtime stack:

- Docker
- Streamlit
- Qdrant
- MariaDB

## Data Availability

Most of the dataset used in this repository are available to download publicly, but the main dataset used for the regression model are not available publicly to comply with Rumah123 terms of service. However, you can use the scrapy project in this repo to build your own dataset or contact the author for more information.

1. [OSM dataset](https://l.kodesiana.com/dataset-rumah-osm)
2. [Bogor shapefile](https://l.kodesiana.com/dataset-rumah-bogorshp)
3. [Qdrant snapshot](https://l.kodesiana.com/dataset-rumah-qdrant_snapshot)
4. House dataset

## Running the Project

There are two way to run this project, (1) using prebuilt Docker image and (2) building from source. In both options, you will need Docker and OpenAI API key to be able to use the "Tanya AI" feature.

### Using Docker

This route is preferred because it comes with a prebuilt dataset and prediction model. You still need to download the Qdrant collection snapshot from the dataset above to be able to use the "Tanya AI" feature.

To start the app, run:

```bash
docker compose -f docker-compose-no-dev.yml up
```

After you have Qdrant server running, download the Qdrant snapshot from links above, create a new collection "bogor_houses", and import the snapshot. Now you can access the web app at http://localhost or if you encountered an error with used port 80, you can change it in the `docker-compose-no-dev.yml` file.

### From Source

If you're going to run this project from source, you will need to follow this overall steps,

1. Create new virtual environment (using `venv` or `conda`)
2. Clone this repository
3. Run the scraping script to build the house price dataset
4. Run the OSM downloader script to download OSM amenities dataset
5. Download Bogor shapefile
6. Run the ETL pipeline to preprocess the raw dataset and create a ready to use dataset for training the machine learning model
7. (optional) Run the training baseline to pick the best algorithm
8. (optional) Run the tuning script to optimize the hyperparameters for random tree or CatBoost models
9. Run the training script to produce a final model
10. Copy the curated dataset and model to `app` directory so the web app can use it (for now the web app only uses the CatBoost model)

#### Preparing the Workspace

The first steps are creating new virtual environment, cloning the repository, and installing the required dependencies.

> Warning: You will also need a X11 server if you are running the modelling script because it uses Matplotlib to create feature importance plot. This is only required if you are running this project in WSL, if you're running this project in Windows or regular Linux, you can ignore this warning.

```bash
# clone repository
git clone https://github.com/fahminlb33/bogor-house-price.git

# create working directories
mkdir -p dataset/shp
mkdir -p dataset/osm
mkdir -p dataset/house
mkdir -p dataset/curated

# option 1: create new conda environment (you can use mamba too)
conda env create -f conda-env.yml
conda activate rumah

# option 2: create new venv
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# export OpenAI key
export OPENAI_API_KEY=sk-xxx
```

#### Obtaining the Dataset

In this step you will build your dataset.

1. the house price dataset (using web scraping)
2. download OSM dataset
3. download Bogor shapefile
4. convert Shapefile to GeoJSON
5. run ETL pipeline
6. build embedding database

```bash
# scrape property listing from Rumah123
make scrape
mv rumah123.json dataset/house/houses-20k.json

# download data from OSM (default to dataset/osm)
python loader/download_overpass.py

# download Bogor shapefile
wget -O bogor.zip https://l.kodesiana.com/dataset-rumah-bogorshp
unzip bogor.zip -d dataset/shp

# convert Shapefile to GeoJSON (default to dataset/bogor.json)
python loader/process_geojson.py

# run ETL pipeline using DuckDB and dbt (default to dataset/curated)
make etl

# create embeddings and store to Qdrant
python ml/create_embeddings.py --dataset dataset/curated/marts_llm_houses.parquet
```

#### Building the Regression Model

For this step you can jump to the training final model, because the default arguments are already tuned to the best parameters described on the research paper.

```bash
# train baseline model to get statistics
python ml/train_baseline.py --dataset dataset/curated/marts_ml_train_sel_all.parquet --run-name all
python ml/train_baseline.py --dataset dataset/curated/marts_ml_train_sel_r.parquet --run-name r
python ml/train_baseline.py --dataset dataset/curated/marts_ml_train_sel_pvalue.parquet --run-name pvalue
python ml/train_baseline.py --dataset dataset/curated/marts_ml_train_sel_manual.parquet --verbose 0 --run-name manual

# hyperparamter search
python ml/tune_random_forest.py --dataset dataset/curated/marts_ml_train_sel_manual.parquet --tracking-url 'http://10.20.20.102:8009'
python ml/tune_catboost.py --dataset dataset/curated/marts_ml_train_sel_manual.parquet --tracking-url 'http://10.20.20.102:8009'

# train final model (the optimal parameter is already set)
python ml/train_random_forest.py --dataset dataset/curated/marts_ml_train_sel_manual.parquet
python ml/train_catboost.py --dataset dataset/curated/marts_ml_train_sel_manual.parquet
```

#### Running the App

Running dbt documentation site,

```bash
# run this in the root directory of the repo
cd transform
dbt docs generate
dbt docs serve
```

Running prediction app from source,

```bash
# run this in the root directory of the repo
cp dataset/bogor.json app/data/bogor.json
cp ml_models/catboost-xxx/model.cbm app/model/house_price_reg.cbm

# start the streamlit app
cd app
cp .env.example .env
nano .env # edit the env values here
streamlit run Dashboard.py
```

Running the prediction app using Docker image,

```bash
docker compose up
```

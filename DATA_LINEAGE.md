# ETL Pipelines

## ETL DAG

```mermaid
flowchart LR
  L0_SCRAPY[L0: scrapy/Rumah123] --> L1_SCRAPY[L1_1_house.py]
  L1_SCRAPY --> L2_REG[L2_1_feature_construction.py]
  L2_REG --> L2_OUTLIERS[L2_2_outlier_removal.py]

  L0_OVERPASS[L0_1_overpass.py] --> L1_OVERPASS[L1_1_overpass.py]

  L1_OVERPASS --> L3_CORR[L3_1_correlations.py]
  L2_OUTLIERS --> L3_CORR
  L2_OUTLIERS --> L3_DASH[L3_2_dashboard.py]
```

- Kota Bogor: https://nominatim.openstreetmap.org/ui/details.html?osmtype=R&osmid=14745927&class=boundary
- Kabupaten Bogor: https://nominatim.openstreetmap.org/ui/details.html?osmtype=R&osmid=14762112&class=boundary

## Data Lineage

```mermaid
flowchart LR
  RAW[Raw: Rumah123/scrapy] --> L0[L0: houses.json]
  L0 --> L11[L1: house_agent_company.parquet]
  L0 --> L12[L1: house_agent.parquet]
  L0 --> L13[L1: house_floor_material.parquet]
  L0 --> L14[L1: house_images.parquet]
  L0 --> L15[L1: house_material.parquet]
  L0 --> L16[L1: house_specs.parquet]
  L0 --> L17[L1: house_tags.parquet]
  L0 --> L18[L1: houses.parquet]
  L13 --> L21[L2: llm.parquet]
  L15 --> L21[L2: llm.parquet]
  L16 --> L21[L2: llm.parquet]
  L17 --> L21[L2: llm.parquet]
  L18 --> L21[L2: llm.parquet]
  L13 --> L22[L2: regression_features.parquet]
  L15 --> L22[L2: regression_features.parquet]
  L16 --> L22[L2: regression_features.parquet]
  L17 --> L22[L2: regression_features.parquet]
  L18 --> L22[L2: regression_features.parquet]
  L11 --> L3[L3: data_mart SQLite]
  L12 --> L3[L3: data_mart SQLite]
  L13 --> L3[L3: data_mart SQLite]
  L14 --> L3[L3: data_mart SQLite]
  L15 --> L3[L3: data_mart SQLite]
  L16 --> L3[L3: data_mart SQLite]
  L17 --> L3[L3: data_mart SQLite]
  L18 --> L3[L3: data_mart SQLite]
```

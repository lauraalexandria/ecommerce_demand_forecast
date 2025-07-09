# ecommerce_demand_forecast

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_orders_dataset.csv) from Kaggle.

## Tools used

* Conda
* Pylint
* pre-commit — pre-commit hooks
* KaggleAPI

## To use the project

Create and activate enviroment
```
conda create -y -n ecommerce-env python=3.9 && conda activate ecommerce-env
```

Install dependencies
```
pip install -r requirements.txt
```

Install pre-commit
```
pre-commit install
```
### Credentials
To change the default behaviour or use a cloud server, copy .env.example to .env with

```
cp .env.example .env
```

And change the default values to your needs. The Kaggle credentials can be view in "Settings" > Down until "API" > "Create New API Token" and the file kaggle.json will be downloaded with the credentials.

### Run commands

1. Extract Data

```
python project/data_extractor.py
```

2. Data Preparation

* Join orders, itemns, products and clients tabelas by keys (`order_id`, `product_id`, `customer_id`);
* **Filters**: Exclude inconsistent dates and select products and cities;
* **Criate new features**: Add hour and adicional temporal information, identify brazilian holidays and detect new clients;
* **Avoid date gaps**: Add missing dates;
* **Weekly values**: Agregate weekly sales by city and nationally;

```
python project/data_preparation.py
```

3. Feature Engineering

Add historical tendencies from existence features.

```
python project/feature_engineering.py
```

### In case of change scripts

It is necessary to run black command before the commits

´´´
black .
´´´

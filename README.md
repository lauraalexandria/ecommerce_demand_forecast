# ecommerce_demand_forecast

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_orders_dataset.csv) from Kaggle.

## Tools used

* Conda
* Pylint
* pre-commit — pre-commit hooks
* KaggleAPI
* GitHub Actions?
* MLFlow
* Makefile
* Docker
* EvidentlyAI
* Pytest


## To use the project

Install docker extensionon VSCode?

## Setup project with Makefile

Create and activate enviroment
```
make -f Makefile.model setup
```

Install dependencies and pre-commit
```
make -f Makefile.model install
```


### Credentials
To change the default behaviour or use a cloud server, copy .env.example to .env with

```
cp .env.example .env
```

And change the default values to your needs. The Kaggle credentials can be view in "Settings" > Down until "API" > "Create New API Token" and the file kaggle.json will be downloaded with the credentials.

### Run commands with Makefile

1. Extract Data
2. Data Preparation
    * Join orders, itemns, products and clients tabelas by keys (`order_id`, `product_id`, `customer_id`);
    * **Filters**: Exclude inconsistent dates and select products and cities;
    * **Criate new features**: Add hour and adicional temporal information, identify brazilian holidays and detect new clients;
    * **Avoid date gaps**: Add missing dates;
    * **Weekly values**: Agregate weekly sales by city and nationally;
3. Feature Engineering
    Add historical tendencies from existence features.
4. Target creation and train/test split
5. Optimization
6. Final model

```
make -f Makefile.model all-model-steps
```

7. Monitor model
```
make -f Makefile.model monitor
```

### Deploy in Docker

1. Build image
```
docker build -t ecommerce_forecast:latest .
```

2. Run model in a container
```
docker run -p 8080:8080 ecommerce_forecast:latest
```


### In case of change scripts

It is necessary to run black command before the commits

´´´
black .
´´´

### Analyze MLFlow

Experiment in ´ecommerce_forecast´ contains model runs and ´ecommerce_forecast_reports´ contains evidently reports.
```
mlflow server --backend-store-uri sqlite:///mlflow.db
```

# E-Commerce Demand Forecast

## Objective

The goal of this project is to develop and build a MLOps pipeline to build and deploy a predictive model to To forecast weekly demand for the 5 best-selling products in an e-commerce store in the city of São Paulo, based on daily sales history. The ultimate goal is to generate accurate forecasts to aid inventory and logistics planning. The dataset used was [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_orders_dataset.csv) from Kaggle.

The project includes:

- **Data Extraction**: Downloding data from source.
- **Data preparation**: Data cleansing, transformation, and feature engineering.
- **Modeling**: Development and training of forecasting models.
- **Evaluation**: Performance analysis and validation of results.
- **Deployment**: Model deployment using Docker.

## Tools used

* Conda - Virtual enviroment
* Pylint - Static code analyser
* pre-commit - pre-commit hooks
* KaggleAPI - Public API for Kaggle's datasets
* GitHub Actions - CI/CD pipeline platform
* MLFlow - Experiment tracker and model register
* Makefile - Plan to create and manage the project
* Docker - Containerization
* EvidentlyAI - ML observability framework
* Pytest - Test framework

## Projet Struture

```
.
├── Dockerfile                     # Configuration file for building the Docker container
├── Makefile.model                 # Makefile for model-related commands (enviroment, training, evaluation, ...)
├── Makefile.prod                  # Makefile for production deployment (Docker, CI/CD)
├── README.md                      # Project documentation (goals, setup, usage, ...)
│
├── data/                          # Data storage
│   ├── processed/                 # Processed/cleaned data (feature-engineered)
│   └── raw/                       # Raw input data (original datasets)
│
├── final_model/                   # Exported final model files (e.g., for deployment)
├── mlflow.db                      # SQLite database for MLflow tracking (experiments, runs)
│
├── notebooks/                     # Jupyter notebooks for exploration
│   └── initial_attempt.ipynb      # Initial EDA, prototyping, or experimentation
│
├── project/                       # Main Python source code
│   ├── __init__.py                # Marks the directory as a Python package
│   ├── catboost_optimization.py   # Hyperparameter tuning for CatBoost models and MLflow registration
│   ├── data_extractor.py          # Data fetching/loading logic
│   ├── data_preparation.py        # Data cleaning and preprocessing
│   ├── feature_engineering.py     # Feature creation/transformation
│   ├── monitor.py                 # Model monitoring with Evidently
│   ├── select_and_register_model.py # Model selection
│   └── temporal_target_and_split.py # Time-based splits and target variable engineering
│
├── reports/                       # Generated reports (e.g., Evidently, model performance)
├── requirements.txt               # Python dependencies (libraries and versions)
│
└── tests/                         # Unit and integration tests
    ├── test_data_preparation.py   # Tests for data preprocessing logic
    └── test_docker_integration.py # Tests for Docker container functionality
```
## Model Access



## How to execute locally

### Setup project with Makefile

1. Create and activate enviroment
```
make -f Makefile.model setup
```

2. Activate enviroment
```
conda activate ecommerce-env
```

3. Install dependencies and pre-commit
```
make -f Makefile.model install
```

### Add your Credentials

Create your own .env based on .env.example
```
cp .env.example .env
```

And change the default values to your needs. The Kaggle credentials can be view in "Settings" > Down until "API" > "Create New API Token" and the file kaggle.json will be downloaded with the credentials.

### Model

This single make command includes:

1. Data Extraction
2. Data Preparation
    * Join orders, itemns, products and clients tables by keys (`order_id`, `product_id`, `customer_id`);
    * **Filters**: Exclude inconsistent dates and select products and cities;
    * **Criate new features**: Add hour and adicional temporal information, identify brazilian holidays and detect new clients;
    * **Avoid date gaps**: Add missing dates;
    * **Weekly values**: Agregate weekly sales by city and nationally;
3. Feature Engineering
   * Add historical tendencies from existence features
4. Target creation and train/test split
5. Catboost Optimization and register in MLflow
6. Select the final model based on the RMSE metric and download it

```
make -f Makefile.model all-model-steps
```

### Monitoring model

Creates a evidently html report about the current and previous predictions. It is also logged in MLFlow.
```
make -f Makefile.model monitor
```

### Open MLFlow

In order to analyze models runs and Evidently report, it is possible to open the MLFlow interface. Experiment ´ecommerce_forecast´ contains model runs and ´ecommerce_forecast_reports´ contains evidently reports.
```
mlflow server --backend-store-uri sqlite:///mlflow.db
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

### Deativate enviroment
```
conda deactivate
```

## To-do list (next improvements)

* [ ] Improve docker deployment/Use cloud
* [ ] Add baseline models as models comparisions
* [ ] Improve MLflow run names and metrics logged (e.g: group by products)
* [ ] Improve pipeline in order to run monthly updates
* [ ] Add alerts in Evidently
* [ ] Improve tests
* [ ] Add external data (e.g: GoogleTrends)
* [ ] Create a streamlit dashboard (?)

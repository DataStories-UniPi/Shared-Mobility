# Shared Mobility Demand Forecasting
- [Shared Mobility Demand Forecasting](#shared-mobility-demand-forecasting)
  - [1. Installation](#1-installation)
  - [2. Documentation:](#2-documentation)
  - [3. Example Usage:](#3-example-usage)


## 1. Installation  

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/DataStories-UniPi/Shared-Mobility.git
   cd Shared-Mobility
   ```

2. **Install dependencies**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

   
## 2. Documentation:
   ```bash
   -h, --help                      show this help message and exit
   --city {Amsterdam, Rotterdam}   Select city to run experiment on                     
   --method {reg, classif}         Select the model variation
   --device {cpu, cuda}            Select whether you want to use CPU or GPU
   ```

## 3. Example Usage:

   In order to train the Regressor model on Rotterdam using GPU, run the following command:
  ```bash
   python src/orchestrator.py --city Rotterdam --method reg --devide cuda 
   ```

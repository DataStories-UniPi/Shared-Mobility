# Shared Mobility Demand Forecasting
- [Shared Mobility Demand Forecasting](#shared-mobility-demand-forecasting)
  - [1. Installation](#1-installation)
    - [1.1. **Clone the repository**:](#11-clone-the-repository)
    - [1.2. **Create and run Docker container**:](#12-create-and-run-docker-container)
    - [1.3. **Specify Parameters for the Trainer**:](#13-specify-parameters-for-the-trainer)
  - [2. Example Usage:](#2-example-usage)
- [Acknowledgments](#acknowledgments)


## 1. Installation  

### 1.1. **Clone the repository**:  
   ```bash
   git clone https://github.com/DataStories-UniPi/Shared-Mobility.git
   cd Shared-Mobility
   ```

### 1.2. **Create and run Docker container**:
   To ensure compatibility, we provide a Dockerfile to run the model inside a container:
   ```bash
   docker build -t shared_mobility .
   docker run -it shared_mobility /bin/bash
   ```
   > **_NOTE:_** If you plan to run this in a linux environment, you need to specify the platform flag `--platform linux/x86_64` during the build
 
### 1.3. **Specify Parameters for the Trainer**:
   In order to centralize the training process we have created a so-called _Orchestrator_. 
   The Area-of-Interest, model variation and device can be selected via the following arguments:
   
   ```bash
   -h, --help                      show this help message and exit
   --city {Amsterdam, Rotterdam}   Select city to run experiment on                     
   --method {reg, classif}         Select the model variation
   --device {cpu, cuda}            Select whether you want to use CPU or GPU
   ```

## 2. Example Usage:

   In order to train the Regressor model on Rotterdam using CPU, run the following command:
  ```bash
   python src/orchestrator.py --city Rotterdam --method reg --device cpu 
   ```

# Acknowledgments
The research work was supported by the **Horizon Europe** R&I programme EMERALDS under the GA _No. 101093051_.
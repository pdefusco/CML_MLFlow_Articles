# CML_MLFlow_Articles

## Objective

Cloudera Machine Learning (CML) is a cloud-native platform that provides a robust environment for developing, deploying, and scaling machine learning models. It offers tools for data scientists, engineers, and analysts to collaborate on AI-driven projects using Python, R, or Scala in secure and isolated workspaces.

MLflow is an open-source platform designed to streamline the machine learning (ML) lifecycle, from experimentation to deployment. It provides tools to manage and track experiments, package code into reproducible runs, and deploy models in various environments.

MLflow is natively supported in Cloudera Machine Learning (CML). In CML, MLflow allows users to log and compare experiment results, organize models, and monitor their performance across different stages of development. Additionally, MLflow's model registry ensures proper versioning and governance, making it easier to deploy models in production within CML's secure environment.

In this set of demo articles and tutorials you will discover how to use popular ML frameworks and tools including SparkML, Apache Iceberg, and XGBoost with MLFlow in CML.

### Content:

1. XGBoost Example
2. Spark & Iceberg Example

##### 1. XGBoost Example

* Open a CML Session with the following settings. ```Editor: Workbench / Edition: Standard / Version: 2024.02 or above / Enable Spark: Spark 3.2 or above / Resource Profile: 2 vCPU & 4GiB Mem & 0 GPUs.```
* In the CML Session, install the requirements.txt with ```pip3 install -r requirements.txt```.
* Open ```00_datagen.py``` and update ```CONNECTION_NAME``` with the Spark Data Connection that is local to your project.
* Run ```00_datagen.py```.
* Run ```01_mlflow_xgboost.py```.

##### 2. Spark & Iceberg Example

* Open a CML Session with the following settings. ```Editor: Workbench / Edition: Standard / Version: 2024.02 or above / Enable Spark: Spark 3.2 or above / Resource Profile: 2 vCPU & 4GiB Mem & 0 GPUs.```
* In the CML Session, install the requirements.txt with ```pip3 install -r requirements.txt```.
* Open the terminal and copy "tmp/spark-executor.json" into your local project home: ```cp /tmp/spark-executor.json spark-executor.json```
* Open ```spark-executor.json``` in the Workbench editor and search for the ```/home/cdsw/``` field. In the corresponding struct, remove the ```"readOnly":true``` key value pair from the file.
* Replace the file under tmp: ```cp spark-executor.json /tmp/spark-executor.json```.
* Now partially run ```01_sparkml_iceberg.py``` until line 104. Notice that the model artifact is logged with the ```mlflow.spark.log_model``` method by referencing the ```dfs_tmpdir``` option. Next, take down the name of the experiment ID and experiment run ID from the output of the ```exp1``` method (or the MLFlow Tracking UI in CML). Finally, update these two values in the model path at line 108, and run the rest of the script.  

### Summary & Next Steps

MLflow's integration into Cloudera Machine Learning (CML) provides a powerful platform for managing the complete machine learning lifecycle, from experimentation to deployment. By leveraging MLflow’s capabilities, CML allows users to track experiments, log metrics, and manage models efficiently, all while ensuring traceability and governance through its model registry feature.

This combination streamlines workflows, enhances collaboration among teams, and supports scalable ML operations across enterprises. It enables data scientists to accelerate the delivery of AI solutions while maintaining control over the entire process.

While you are exploring MLFlow in Cloudera Machine Learning you might find the following articles and blogs useful.

- **Introducing MLOps and SDX for Models in Cloudera Machine Learning**: This blog explores the integration of MLflow into CML for managing machine learning workflows, focusing on experiment tracking, model development, and deployment. It discusses how Cloudera enables data scientists to streamline their processes while maintaining governance through MLflow's capabilities. [Read more here](https://blog.cloudera.com/introducing-mlops-and-sdx-for-models-in-cloudera-machine-learning)【13†source】.

- **Running Experiments using MLflow in CML**: This documentation provides a step-by-step guide to running MLflow experiments in Cloudera Machine Learning, detailing how to track parameters, metrics, and artifacts in your workflows. [Check it out here](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-exp-v2-run-exp-mlflow.html)【15†source】.

- **CML’s New Experiments Feature Powered by MLflow**: This community article explains how MLflow is integrated into Cloudera Machine Learning's new experiment tracking feature, enabling data scientists to log experiment results, compare models, and improve their workflow efficiency. [Learn more](https://community.cloudera.com/t5/Community-Articles/CML-s-new-Experiments-feature-powered-by-MLflow-enables-data/ta-p/358684)【14†source】.

- **Using Model Registry with MLflow in CML**: This blog covers how Cloudera's Model Registry feature works in tandem with MLflow, allowing teams to version and deploy models while maintaining traceability and automation. [Read the full blog](https://community.cloudera.com/t5/Community-Articles/How-to-use-Model-Registry-on-Cloudera-Machine-Learning/ta-p/379812)【16†source】.

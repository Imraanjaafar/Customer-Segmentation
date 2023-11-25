# Customer-Segmentation

##  Build a deep learning model for anticipating campaign outcomes
Customer segmentation involves categorizing individuals based on age, gender, profession, and interests, aiding companies in understanding customer needs. This insight informs targeted marketing strategies, optimizing profitability. A recent decline in a bank's revenue prompted an investigation, revealing reduced client deposits as the main cause. To address this, the bank initiated marketing campaigns to encourage more deposits, aiming to enhance overall satisfaction. Essential aspects of these campaigns include customer segmentation and promotional strategy. As a data analyst and deep learning engineer, your task involves using a dataset detailing phone marketing campaigns to develop a predictive deep learning model for campaign outcomes, contributing to the bank's growth. The dataset used in this project are from https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon

## Directory Structure
- [Ahmad_Imran_CustomerSegmentation.py](https://github.com/Imraanjaafar/Customer-Segmentation/blob/main/Ahmad_Imran_CustomerSegmentation.py)
  Here's a generalized explanation of the code:

### Import Libraries:

- The script begins by importing necessary libraries such as os, pandas, tensorflow, matplotlib, scipy, and seaborn.

### Load Data:

- The script loads a CSV file (train.csv) into a Pandas DataFrame.

### Data Inspection and Preprocessing:

- The script defines a class BankMarketingData to encapsulate data loading.
- Various operations are performed to inspect and preprocess the data, including checking for null values, handling duplicates, and filling missing values.

### Data Visualization:

= Several data visualizations are created using Matplotlib and Seaborn, including box plots, histograms, and Q-Q plots.

### Outlier Removal:

- Outliers are removed from the 'num_contacts_in_campaign' column, and the results are visualized before and after the removal.

### Correlation Analysis:

- Correlation matrices are created and visualized using a heatmap.

### Categorical Data Handling:

- Categorical columns are encoded using ordinal encoding, and data distributions for categorical features are visualized.

### One-Hot Encoding:

- The target variable 'term_deposit_subscribed' is one-hot encoded.

### Data Splitting:

- The data is split into training and testing sets.

### Neural Network Model Building:

- A neural network model is defined using the TensorFlow Keras API.
- The model architecture includes input, hidden, and output layers with activation functions like 'relu' and 'softmax'.
- Batch normalization and dropout layers are added to improve model generalization.

### Model Compilation:

- The model is compiled with the Adam optimizer and categorical cross-entropy loss.

### Callbacks:

- Callbacks such as EarlyStopping and TensorBoard are defined for monitoring training progress.

### Model Training:

- The model is trained on the training data with a specified number of epochs and batch size.

### Model Evaluation:

- The trained model is evaluated on the test set, and metrics like loss and accuracy are printed.

### Model Saving:

- The trained model is saved in the 'models' directory.


## Results

- The model demonstrated its effectiveness in customer segmentation by achieving an 89.00% accuracy on the test set.


## Model Architecture

![model](https://github.com/Imraanjaafar/Customer-Segmentation/assets/151133555/b8245d7d-e5d3-4036-95bf-b01ff8359baa)




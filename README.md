# Customer-Segmentation

##  Deep learning 

This assessment necessitated  the creation of a deep learning model for forecasting the campaign's results.

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


## Model Architecture



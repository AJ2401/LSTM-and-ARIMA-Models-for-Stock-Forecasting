# Using LSTM and ARIMA Models for Stock Forecasting
The LSTM model serves as the primary forecasting tool, leveraging its ability to capture long-term dependencies in sequential data. However, recognizing that even sophisticated models like LSTM can have prediction biases, an ARIMA model is employed to estimate and correct these errors. By doing so, the system harnesses the strengths of both models: LSTM's deep learning capabilities for handling complex patterns and ARIMA's effectiveness in modeling time series data.

The repository includes a detailed script that outlines the entire process, from data loading and preprocessing to model training and evaluation. The data_loader function sets the stage, preparing the dataset for analysis. It's followed by a series of plotting functions that visualize various aspects of the data, such as raw time series, training versus testing sets, and prediction errors.

The LSTM model's architecture is defined with several layers, including LSTM and Dense layers, and the model is trained using the historical closing prices of financial assets. After training, the model's predictions are plotted against the actual values to visualize the performance.

The ARIMA model then steps in to calculate the error of the LSTM's predictions. These error estimates are subsequently used to adjust the LSTM predictions, resulting in a final, corrected output. This final prediction is believed to be more accurate and is visualized alongside the actual data for evaluation.

Performance metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) are calculated to quantify the accuracy of the models. The repository captures these metrics in a structured format, allowing for clear interpretation of the model's effectiveness.


## Data

Time-series data is a sequential collection of data points recorded at specific time intervals. In financial markets, time-series data primarily consists of stock prices, trading volumes, and various financial indicators gathered at regular time intervals. The significance of time-series data lies in its chronological order, a fundamental aspect that enables the identification of trends, cycles, and patterns critical for forecasting.


## Long Short-Term Memory (LSTM):
### Sequential Data Analysis a Leap Forward with LSTM Networks

### Introduction :
The inception of Long Short-Term Memory (LSTM) networks marked a pivotal advancement in the field of sequential data analysis. These networks, a specialised evolution of Recurrent Neural Network (RNN) architectures, emerged to address the challenge of preserving information over extended sequences – a hurdle where traditional RNNs faltered due to the vanishing gradient dilemma. LSTMs were ingeniously crafted to retain critical data across long intervals, ensuring that pivotal past information influences future decisions.

### Decoding LSTM Mechanisms : 
In my program, I have utilised TensorFlow to construct and train an LSTM-based model for a specific task, likely related to time series forecasting. Let's break down how each of the LSTM components corresponds to my program:

### Memory Cell (LSTM Cell) :

In my program, the memory cell is represented implicitly by the LSTM layer I've added using tf.keras.layers.LSTM(number_nodes, input_shape=(n, 1)). This LSTM layer acts as the memory cell of the network.
The memory cell's purpose in my program is to capture and retain information over extended sequences. It is responsible for learning and remembering patterns and dependencies in the input time series data (middle_data) over time.

### Input Gate:

The input gate is a crucial part of an LSTM unit that regulates what information should be added to the memory cell. It uses a sigmoid function to control the flow of input information and employs a hyperbolic tangent (tanh) function to create a vector of values ranging from -1 to +1.

In my program, the input gate is implicitly implemented by the LSTM layer (tf.keras.layers.LSTM) within TensorFlow. The LSTM layer manages the flow of input information, determines what information should be stored in its cell state, and applies appropriate weightings using sigmoid and tanh functions.

### Forget Gate:

The forget gate is responsible for deciding which information in the memory cell should be discarded. It employs a sigmoid function to assess the importance of each piece of information in the current memory state.
In my program, the forget gate's functionality is automatically handled by the LSTM layer. It learns to decide which information from the previous memory state should be forgotten or retained based on the patterns and dependencies it identifies in the input data.

### Output Gate:

The output gate extracts valuable information from the memory cell to produce the final output. It combines the content of the memory cell with the input data, employing both tanh and sigmoid functions to regulate and filter the information before presenting it as the output.
In my program, the output gate's operations are also encapsulated within the LSTM layer. It takes the current memory state and the input data to produce an output that is used for making predictions.


## ARIMA: 
### The Linear Approach to Time-Series Forecasting

### Introduction to the ARIMA Model:
The Autoregressive Integrated Moving Average (ARIMA) model stands as a fundamental pillar within the realm of statistical time-series analysis. Its inception by Box and Jenkins in the early 1970s brought forth a powerful framework that amalgamates autoregressive (AR) and moving average (MA) elements, all while incorporating differencing to stabilise the time-series (the "I" in ARIMA). ARIMA models are celebrated for their simplicity and efficacy in modelling an extensive array of time-series data, notably for their proficiency in capturing linear relationships.

- Error Mining with ARIMA : After LSTM's predictions, the program calls on ARIMA to refine these forecasts. The Error_Evaluation function comes into play here, extracting the difference between the predicted and actual prices—essentially capturing the LSTM's predictive shortcomings.

- ARIMA's Calibration : With the error data in hand, the ARIMA_Model function is invoked, wielding the ARIMA model as a fine brush to paint over the imperfections of the LSTM's initial output. The ARIMA model is trained on these residuals, learning to anticipate the LSTM's prediction patterns and, more importantly, its prediction errors.

- Synthesis of Predictions : The Final_Predictions function represents the judgement of the program's operations. It does not merely output raw predictions but synthesises the LSTM's foresight with ARIMA's insights, producing a final prediction that encapsulates the strengths of both models.


## Integrating LSTM and ARIMA

The integration of LSTM and ARIMA models presents a compelling hybrid approach to time-series forecasting. This methodology draws on the strengths of both models: LSTMs are capable of capturing complex non-linear patterns, while ARIMA excels at modelling the linear aspects of a time-series. By combining these two, one can potentially mitigate their individual weaknesses and enhance the overall predictive power.

### Analysis of Combined Model Predictions : 
Upon integrating LSTM and ARIMA, the model becomes robust against the volatility and unpredictability of financial time-series data. The predictions from the LSTM can be refined by the ARIMA model's error correction mechanism, which adds another layer of sophistication to the forecasts.

### Comparative Analysis: LSTM vs. LSTM+ARIMA vs. Actual Values :

The predictions from LSTM, the hybrid LSTM+ARIMA model, and the actual values, several insights emerge. The LSTM model may capture the momentum and direction of stock prices effectively, but it might struggle with precision due to its sensitivity to recent data. The ARIMA model, conversely, may lag in capturing sudden market shifts but provides a smoothed forecast that averages out noise.

The hybrid model aims to balance these aspects. The LSTM component may anticipate a trend based on recent patterns, and the ARIMA part can adjust this forecast by considering the broader historical context. The final predictions, ideally, are more aligned with the actual values than either model could achieve on its own.


Implementation of the Program : 

### Function Definition and Working

#### `data_loader()`

**Purpose:**
The `data_loader` function is designed to load financial time-series data from a CSV file and prepare it as a DataFrame formatted for time series analysis.

**Input:**
The function takes no parameters but relies on a globally defined `Filename_address` variable that contains the path to the CSV file.

**Processing Elements:**
1. **Pandas Library:** Utilized for its powerful data manipulation capabilities, particularly for reading CSV files and handling time series data.
2. **Global Variables:** It uses the `Filename_address` to locate the CSV file.
3. **DataFrame Operations:**
   - `pd.read_csv`: Reads the CSV file into a DataFrame, with the 'Date' column set as the index and parsed as datetime objects for time series analysis.
   - `dropna`: Removes any rows with missing values to ensure the integrity of the time series data.


**Output:**
The function returns a `DataFrame` object containing the clean, time-indexed financial data.


### Pseudo Code Algorithm

```
Function data_loader
    Define column names as ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
    Load CSV file from 'Filename_address' into a DataFrame with 'Date' as index
    Set DataFrame columns to the defined column names
    Drop any rows with missing values
    Print the shape of the DataFrame
    Print the first few rows of the DataFrame
    Return the cleaned DataFrame
EndFunction
```

### Flow of the Program for `data_loader()`

1. Initialize the column names for the financial data.
2. Use the Pandas function `read_csv` to read the data from the CSV file specified by the `Filename_address`.
3. Set the index of the DataFrame to the 'Date' column, which is parsed as datetime.
4. Assign the predefined column names to the DataFrame to maintain consistency.
5. Remove any rows with missing data to ensure the data quality for subsequent analysis.

### Function Definition and Working

#### `plot_predictions(train, predictions, title)`

**Purpose:**
The `plot_predictions` function is designed to visualize the actual vs. predicted financial time-series data. It generates a plot that overlays the predicted values over the actual values, allowing for a visual comparison.

**Input:**
- `train`: A pandas Series or DataFrame containing the actual values indexed by date.
- `predictions`: A pandas Series or DataFrame containing the predicted values, expected to be of the same length and with the same index as `train`.
- `title`: A string representing the title of the plot, which will also be used in naming the saved plot file.

**Processing Elements:**
1. **Matplotlib Library:** Used for creating visualizations.
2. **Global Variables:** Utilizes `Output_address` to determine the save path for the plot image.

**Output:**
- The function saves a .jpg image file of the plot to the location specified by `Output_address` with the given `title` as its name.
- No value is returned by the function.


### Pseudo Code Algorithm

```
Function plot_predictions with parameters: train, predictions, title
    Initialize a new figure with specified dimensions (10x5 inches)
    Plot the 'train' data with the index on the x-axis and values on the y-axis, labeled as 'Actual'
    Plot the 'predictions' data on the same axes, labeled as 'Predicted' in red color
    Set the title of the plot
    Set the x-axis label as 'Date'
    Set the y-axis label as 'Close-Price'
    Concatenate the `Output_address` with the `title` and ".jpg" to form the file path
    Save the figure to the file path
EndFunction
```

### Flow of the Program for `plot_predictions()`

1. Start by creating a new figure with the defined size.
2. Plot the actual values (`train`) against their date index, labeling this line as 'Actual'.
3. Plot the predicted values (`predictions`) on the same plot, using a different color and labeling it 'Predicted'.
4. Assign the provided `title` to the plot.
5. Label the x-axis as 'Date' and the y-axis as 'Close-Price' to indicate what the axes represent.
6. Combine the `Output_address` directory path with the `title` of the plot to create the full file path for saving.
7. Save the figure as a .jpg file at the determined file path.
8. The plot is now saved to the local file system, and the function terminates without returning any value.



### Function Definition and Working

#### `plot_train_test(train, test)`

**Purpose:**  
The `plot_train_test` function generates a plot to visualize the partition of financial time-series data into training and testing sets. This visual aid is important to verify the partitioning and observe the continuity and potential discrepancies between the train and test sets.

**Input:**
- `train`: A pandas Series or DataFrame containing the training set data, indexed by date.
- `test`: A pandas Series or DataFrame containing the testing set data, indexed by date.

**Processing Elements:**
1. **Matplotlib Library:** Used for creating and saving the plot.
2. **Global Variables:** The function uses `Output_address` for determining where to save the output image.

**Output:**
- The function outputs a plot saved as a .jpg file to the location specified by `Output_address`. The plot displays the training and testing data series.


### Pseudo Code Algorithm

```
Function plot_train_test with parameters: train, test
    Initialize a new figure with a size of 10x5 inches
    Plot the 'train' series against its index with a label 'Train Set'
    Plot the 'test' series against its index with a label 'Test Set' and set the color to orange
    Set the title of the plot to 'Train and Test Data'
    Set the x-axis label to 'Date'
    Set the y-axis label to 'Close Price'
    Concatenate `Output_address` with the filename ' Train and Test Data .jpg'
    Save the figure to the specified address
EndFunction
```

### Flow of the Program for `plot_train_test()`

1. Begin by initiating a new figure for plotting with specified dimensions (10x5 inches).
2. Plot the training dataset (`train`) on the figure, with dates on the x-axis and training data values on the y-axis, labeling it as 'Train Set'.
3. Plot the testing dataset (`test`) on the same figure, with dates on the x-axis and testing data values on the y-axis, labeling it as 'Test Set' and using a distinct orange color for differentiation.
4. Title the plot 'Train and Test Data' to describe the plotted data.
5. Label the x-axis as 'Date' to indicate the time component and the y-axis as 'Close Price' to denote the financial metric plotted.
6. Construct the file path for saving the plot by combining `Output_address` with the designated file name ' Train and Test Data .jpg'.
7. Save the plot to the constructed file path.
8. The function concludes after saving the plot, and it does not return any values.


### Function Definition and Working

#### `plot_prediction_errors(errors)`

**Purpose:**  
The `plot_prediction_errors` function is used to visualize the errors over time between actual and predicted values in a time series forecasting model. This can help in identifying patterns or biases in the prediction errors.

**Input:**
- `errors`: A list or pandas Series containing the prediction errors, typically calculated as the difference between actual and predicted values.

**Processing Elements:**
1. **Matplotlib Library:** This function utilizes Matplotlib to create and save a visualization plot of the prediction errors.
2. **Global Variables:** `Output_address` is used to determine where the plot image will be saved.

**Output:**
- The function saves a .jpg file of the error plot to the directory specified by `Output_address`.


### Pseudo Code Algorithm

```
Function plot_prediction_errors with parameter: errors
    Initialize a new figure with a size of 10x5 inches
    Plot 'errors' with labeling as 'Prediction Errors'
    Set the title of the plot to 'Prediction Errors over Time'
    Set the x-axis label to 'Time Step'
    Set the y-axis label to 'Error'
    Create a legend for the plot
    Form the save address by concatenating `Output_address` with ' Prediction Errors over Time .jpg'
    Save the figure to the address
EndFunction
```

### Flow of the Program for `plot_prediction_errors()`

1. Initiate a new figure with the specified dimensions for the plot.
2. Plot the errors provided by the `errors` parameter against their corresponding time step.
3. Title the plot 'Prediction Errors over Time' to accurately reflect the data being visualized.
4. Label the x-axis as 'Time Step' to represent the sequential nature of the data points.
5. Label the y-axis as 'Error' to represent the magnitude of the prediction errors.
6. Add a legend to the plot for clarity, which describes the data series plotted.
7. Construct the full file path where the plot will be saved by appending ' Prediction Errors over Time .jpg' to the `Output_address`.
8. Save the plot to the specified file path.
9. The function completes its execution after the plot is saved, without returning any values.

### Function Definition and Working

#### `plot_final_predictions(test, final_predictions)`

**Purpose:**  
`plot_final_predictions` is designed to create a visualization comparing the actual values from the test dataset with the final corrected predictions. This helps to assess the accuracy and effectiveness of the error correction applied to the predictive model.

**Input:**
- `test`: A pandas Series or DataFrame containing the test set data, indexed by date.
- `final_predictions`: A pandas Series or DataFrame of the same length and with the same index as `test` containing the final predictions after error correction.

**Processing Elements:**
1. **Matplotlib Library:** It is utilized for plotting and saving the comparison plot.
2. **Global Variables:** The function requires `Output_address` to define the path where the plot image will be saved.

**Output:**
- The function outputs a plot saved as a .jpg file to the location determined by `Output_address`. The plot displays the actual values and the corrected predictions.


### Pseudo Code Algorithm

```
Function plot_final_predictions with parameters: test, final_predictions
    Initialize a new figure with a size of 10x5 inches
    Plot the 'test' series against its index with a label 'Actual'
    Plot the 'final_predictions' series against the same index with a label 'Corrected Prediction' in green color
    Set the title of the plot to 'Final Predictions with Error Correction'
    Set the x-axis label to 'Date'
    Set the y-axis label to 'Close Price'
    Create a legend for the plot
    Form the save address by concatenating `Output_address` with the file name ' Final Predictions with Error Correction .jpg'
    Save the figure to the constructed address
EndFunction
```

### Flow of the Program for `plot_final_predictions()`

1. Begin by initiating a new plotting figure with the given dimensions.
2. Plot the actual test data (`test`) with the date index on the x-axis and close prices on the y-axis, labeled as 'Actual'.
3. Plot the final corrected predictions (`final_predictions`) on the same axes, labeling it as 'Corrected Prediction' and using green color for distinction.
4. Title the plot 'Final Predictions with Error Correction' to describe its purpose.
5. Label the x-axis 'Date' and the y-axis 'Close Price' to indicate what the plot represents.
6. Add a legend to the plot to identify the data series.
7. Construct the file path for saving the plot by combining `Output_address` with the file name ' Final Predictions with Error Correction .jpg'.
8. Save the plot to the determined file path.
9. The function concludes after saving the plot, and it does not return any value.

### Function Definition and Working

#### `plot_accuracy(mse, rmse, mae)`

**Purpose:**  
The `plot_accuracy` function generates a bar chart to visually represent the accuracy metrics of a predictive model. These metrics typically include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

**Input:**
- `mse`: A numerical value representing the Mean Squared Error.
- `rmse`: A numerical value representing the Root Mean Squared Error.
- `mae`: A numerical value representing the Mean Absolute Error.

**Processing Elements:**
1. **Matplotlib Library:** Used for plotting and saving the accuracy metrics as a bar chart.
2. **Global Variables:** The function uses `Output_address` to determine the directory path where the plot image will be saved.

**Output:**
- The function outputs a bar chart saved as a .jpg file to the directory specified by `Output_address`.

### Pseudo Code Algorithm

```
Function plot_accuracy with parameters: mse, rmse, mae
    Define a list 'metrics' with the values 'MSE', 'RMSE', 'MAE'
    Define a list 'values' with the input parameters mse, rmse, mae
    Initialize a new figure with a size of 10x5 inches
    Plot a bar chart with 'metrics' as the x-axis and 'values' as the heights of the bars
    Assign different colors to each bar for distinction
    Set the title of the plot to 'Model Accuracy Metrics'
    Form the save address by concatenating `Output_address` with the file name ' Model Accuracy Metrics .jpg'
    Save the figure to the specified address
EndFunction
```

### Flow of the Program for `plot_accuracy()`

1. Define the names of the metrics to be plotted (MSE, RMSE, MAE) in a list.
2. Gather the provided accuracy metric values into a list corresponding to the metric names.
3. Initialize a new plotting figure with predetermined dimensions (10x5 inches).
4. Create a bar chart with the metric names on the x-axis and their corresponding values as the heights of the bars, with each bar colored differently for easy distinction.
5. Title the plot 'Model Accuracy Metrics' to clearly indicate what the chart represents.
6. Determine the file path for saving the plot by appending ' Model Accuracy Metrics .jpg' to the `Output_address`.
7. Save the bar chart to the constructed file path.
8. The function ends after the bar chart is saved and does not return any values.

### Function Definition and Working

#### `plot_arima_accuracy(mse, rmse, mae)`

**Purpose:**  
The `plot_arima_accuracy` function visualizes the accuracy metrics specific to an ARIMA model using a bar chart. This visualization assists in the evaluation of the model's performance by representing Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) as bar heights.

**Input:**
- `mse`: A numeric value indicating the Mean Squared Error.
- `rmse`: A numeric value indicating the Root Mean Squared Error.
- `mae`: A numeric value indicating the Mean Absolute Error.

**Processing Elements:**
1. **Matplotlib Library:** Employs matplotlib to create and save a bar chart.
2. **Global Variables:** The function utilizes `Output_address` for the path where the bar chart will be saved.

**Output:**
- This function outputs a bar chart saved as a .jpg file in the directory specified by `Output_address`.

### Pseudo Code Algorithm

```
Function plot_arima_accuracy with parameters: mse, rmse, mae
    Define a list 'metrics' with elements 'MSE', 'RMSE', 'MAE'
    Define a list 'values' with the input parameters mse, rmse, mae
    Initialize a new figure with dimensions of 10 by 5 inches
    Create a bar chart with 'metrics' on the x-axis and 'values' as the bar heights
    Assign specific colors to each bar (blue for MSE, orange for RMSE, green for MAE)
    Set the chart title to 'ARIMA Model Accuracy Metrics'
    Determine the save address by concatenating `Output_address` with ' Model Accuracy Metrics .jpg'
    Save the figure to the defined address
EndFunction
```

### Flow of the Program for `plot_arima_accuracy()`

1. Initialize a list called `metrics` with the names of the accuracy metrics to be displayed.
2. Create a list called `values` containing the values of MSE, RMSE, and MAE passed to the function.
3. Begin a new plot with a figure size set to 10x5 inches.
4. Plot a bar chart where the x-axis contains the metric names from `metrics` and the y-axis corresponds to their respective values from `values`.
5. Assign a distinct color to each bar to visually differentiate between the metrics.
6. Title the plot 'ARIMA Model Accuracy Metrics' to clearly convey the plot's focus.
7. Formulate the full file path for saving the chart by appending ' Model Accuracy Metrics .jpg' to the `Output_address`.
8. Save the bar chart to the file path that was created.
9. The function terminates after the plot is saved, without returning any value.

### Function Definition and Working

#### `data_allocation(data)`

**Purpose:**  
The `data_allocation` function is tasked with partitioning a given dataset into training and testing sets for model development and evaluation. This split is essential for assessing the model's performance on unseen data.

**Input:**
- `data`: A pandas DataFrame that contains the time series data with one of the columns being `close`, representing the closing price which is typically used in financial time series forecasting.

**Processing Elements:**
1. **Global Variables:**
   - `days`: The number of entries from the end of the dataset to be allocated to the test set.
   - `close`: A string that denotes the column name for the closing prices in the `data` DataFrame.

**Output:**
- `train`: A pandas Series or DataFrame containing the training set data.
- `test`: A pandas Series or DataFrame containing the testing set data.

### Pseudo Code Algorithm

```
Function data_allocation with parameter: data
    Calculate train_len_val by subtracting the number of days (global variable) from the length of the data
    Split the 'data' into 'train' and 'test' sets by slicing:
        'train' contains all entries from start up to train_len_val
        'test' contains all entries from train_len_val to the end
    Print the training set and its size
    Print the testing set and its size
    Return the 'train' and 'test' sets
EndFunction
```

### Flow of the Program for `data_allocation()`

1. Determine the length of the training set by subtracting the global variable `days` from the total length of the dataset.
2. Allocate the first segment of the dataset up to the determined length to the training set.
3. Allocate the remaining segment from the determined length to the end of the dataset to the testing set.
4. Print a descriptive message followed by the training set and its size to provide an immediate visual confirmation of the data partitioning.
5. Print a descriptive message followed by the testing set and its size for the same reasons as above.
6. Return both the training set and the testing set to be used in subsequent stages of the model development and evaluation process.

### Function Definition and Working

#### `apply_transform(data, n)`

**Purpose:**  
The `apply_transform` function is designed to transform time series data into a format suitable for training LSTM (Long Short-Term Memory) networks. The transformation involves creating sequences of `n` previous data points (lags) to predict the next value.

**Input:**
- `data`: A pandas Series or numpy array containing the time series data.
- `n`: An integer that defines the number of lags, i.e., the size of the input sequence for the LSTM model.

**Processing Elements:**
1. **NumPy Library:** Used for numerical operations and to transform the list of sequences into a numpy array suitable for the LSTM input.
2. **List Comprehension:** Constructs the sequences of lags (input data) and the target values (what the model will learn to predict).

**Output:**
- `middle_data`: A numpy array of shape `(number of sequences, n, 1)`, where each sequence is a sliding window of `n` lagged values from the `data`.
- `target_data`: A numpy array containing the target values corresponding to each sequence in `middle_data`.

### Pseudo Code Algorithm

```
Function apply_transform with parameters: data, n
    Initialize an empty list called 'middle_data'
    Initialize an empty list called 'target_data'
    Loop over the data starting from index n to the end of the data:
        Extract a sequence of 'n' values from 'data' ending at the current index
        Append the sequence to 'middle_data'
        Append the current value of 'data' to 'target_data'
    Convert 'middle_data' into a numpy array and reshape it to (len(middle_data), n, 1)
    Convert 'target_data' into a numpy array
    Return 'middle_data' and 'target_data'
EndFunction
```

### Flow of the Program for `apply_transform()`

1. Initialize two empty lists: `middle_data` for storing the input sequences and `target_data` for the corresponding target values.
2. Iterate over the `data` series starting from the `n`th element to the end.
3. For each iteration, extract a sequence of `n` values from the `data` series leading up to the current index and append this sequence to `middle_data`.
4. Append the value at the current index of the `data` series to `target_data` as the target value for the previously extracted sequence.
5. After the loop, convert `middle_data` into a numpy array and reshape it to have the dimensions suitable for LSTM input, which is `(number of sequences, n, 1)`.
6. Convert `target_data` into a numpy array without reshaping since it represents the target values.
7. Return the `middle_data` and `target_data` arrays for use in training the LSTM model.

### Function Definition and Working

#### `LSTM(train, n, number_nodes, learning_rate, epochs, batch_size)`

**Purpose:**  
The `LSTM` function builds, compiles, and trains a Long Short-Term Memory (LSTM) neural network model using the provided time series training data. The model aims to predict future values in the series based on the input sequences of historical data.

**Input:**
- `train`: A pandas Series or numpy array containing the time series training data.
- `n`: An integer defining the number of lagged data points to use as input for the LSTM model.
- `number_nodes`: The number of neurons in each LSTM and Dense layer of the neural network.
- `learning_rate`: The learning rate for the optimizer during training.
- `epochs`: The number of epochs to train the model.
- `batch_size`: The number of samples per gradient update during training.

**Processing Elements:**
1. **TensorFlow and Keras:** Utilized for creating the LSTM model, compiling it, and fitting it to the training data.
2. **apply_transform Function:** Called to transform the training data into sequences suitable for LSTM input.
3. **Sequential Model API:** Used for stacking layers to build the LSTM model.
4. **Adam Optimizer:** An algorithm for first-order gradient-based optimization of stochastic objective functions.

**Output:**
- `model`: The trained Keras Sequential LSTM model.
- `history`: A record of training loss and accuracy values at successive epochs.
- `full_predictions`: The model's predictions for the input data used during training.

### Pseudo Code Algorithm

```
Function LSTM with parameters: train, n, number_nodes, learning_rate, epochs, batch_size
    Transform 'train' data into sequences and targets using apply_transform function
    Initialize a Sequential LSTM model
        Add Input layer with shape (n,1)
        Add LSTM layer with 'number_nodes' neurons
        Add two Dense layers each with 'number_nodes' neurons and 'relu' activation
        Add a Dense output layer with a single neuron
    Compile the model with 'mse' loss function, Adam optimizer with 'learning_rate', and 'mean_absolute_error' metric
    Fit the model to 'middle_data' and 'target_data' for 'epochs' with 'batch_size', without verbosity
    Predict on 'middle_data' to obtain full predictions
    Return the model, training history, and full predictions
EndFunction
```

### Flow of the Program for `LSTM()`

1. Call `apply_transform` with the training data `train` and the lag value `n` to prepare the input and target data for the LSTM.
2. Define the LSTM model architecture using the Sequential API from Keras with an input layer, LSTM layer, two dense layers, and an output layer.
3. Compile the LSTM model with the mean squared error loss function, Adam optimizer with the specified learning rate, and mean absolute error as a performance metric.
4. Train the model on the transformed data for the given number of epochs and batch size.
5. After training, use the model to predict on the input data to get the full set of predictions.
6. Output the trained model, the history of its performance over the epochs, and the full predictions array.

### Function Definition and Working

#### `calculate_accuracy(true_values, predictions)`

**Purpose:**  
The function `calculate_accuracy` computes common statistical accuracy metrics to evaluate the performance of regression models, specifically Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

**Input:**
- `true_values`: An array-like structure, typically a numpy array or pandas Series, that contains the actual observed values.
- `predictions`: An array-like structure with the predicted values, expected to be of the same length as `true_values`.

**Processing Elements:**
1. **Mean Squared Error (MSE):** This metric measures the average of the squares of the errors, i.e., the average squared difference between the estimated values and the actual value.
2. **Root Mean Squared Error (RMSE):** It is the square root of the MSE and measures the standard deviation of the residuals.
3. **Mean Absolute Error (MAE):** This metric measures the average magnitude of the errors in a set of predictions, without considering their direction.

**Output:**
- `mse`: A float representing the Mean Squared Error.
- `rmse`: A float representing the Root Mean Squared Error.
- `mae`: A float representing the Mean Absolute Error.

### Pseudo Code Algorithm

```
Function calculate_accuracy with parameters: true_values, predictions
    Calculate MSE by taking the mean of the squared differences between true_values and predictions
    Calculate RMSE by taking the square root of MSE
    Calculate MAE by taking the mean of the absolute differences between true_values and predictions
    Return mse, rmse, mae
EndFunction
```

### Flow of the Program for `calculate_accuracy()`

1. Utilize the `mean_squared_error` function from sklearn.metrics to calculate the MSE between the `true_values` and `predictions`.
2. Compute the RMSE by taking the square root of the MSE using numpy's `sqrt` function.
3. Calculate the MAE using the `mean_absolute_error` function from sklearn.metrics.
4. Return the computed values of MSE, RMSE, and MAE to be used as accuracy metrics for the model evaluation.


### Function Definition and Working

#### `Error_Evaluation(train_data, predict_train_data, n)`

**Purpose:**  
The `Error_Evaluation` function is designed to calculate the errors between the actual training data and the predictions made by the LSTM model. This can be used for further analysis of the model's performance and error correction.

**Input:**
- `train_data`: A pandas Series or numpy array containing the actual observed training values.
- `predict_train_data`: A pandas Series or numpy array containing the predicted values obtained from the LSTM model, expected to be of the same length as `train_data` after accounting for the lag `n`.
- `n`: An integer representing the number of lagged observations used in the LSTM model (the size of the input sequence).

**Processing Elements:**
1. **List Comprehension:** Iterates through the predicted data to compute the difference with the actual data, point by point.

**Output:**
- `errors`: A list of error values representing the difference between the actual and predicted values.

### Pseudo Code Algorithm

```
Function Error_Evaluation with parameters: train_data, predict_train_data, n
    Initialize an empty list called 'errors'
    Loop through the indices of predict_train_data:
        Calculate the error at each point as the difference between the actual value (train_data at index n+i) and the predicted value (predict_train_data at index i)
        Append the error to the 'errors' list
    Return the 'errors' list
EndFunction
```

### Flow of the Program for `Error_Evaluation()`

1. Initialize an empty list to store the error values.
2. Iterate over the predicted training data.
3. For each predicted value, calculate the error by subtracting the predicted value from the actual value (considering the lag `n`).
4. Store each error value in the list.
5. Return the complete list of errors after the iteration is finished. This list can be used to analyze the distribution and pattern of errors made by the model during training.

### Function Definition and Working

#### `Parameter_calculation(data)`

**Purpose:**  
The `Parameter_calculation` function aims to determine the optimal parameters for an ARIMA (Autoregressive Integrated Moving Average) model using the given time series data. It also generates plots for the Autocorrelation Function (ACF) and the Partial Autocorrelation Function (PACF), which are helpful for identifying the ARIMA model's parameters.

**Input:**
- `data`: A pandas Series or numpy array containing the time series data.

**Processing Elements:**
1. **auto_arima from pmdarima:** This is a function that automates the process of ARIMA modeling, including the selection of optimal parameters.
2. **plot_acf from statsmodels:** Generates an ACF plot, which is used to identify the number of MA (Moving Average) terms.
3. **plot_pacf from statsmodels:** Generates a PACF plot, which is used to identify the number of AR (Autoregressive) terms.
4. **Global Variables:**
   - `lag`: Used to set the number of lags in the ACF and PACF plots.
   - `Output_address`: Used to specify the directory path where the ACF and PACF plot images will be saved.

**Output:**
- `ord`: A tuple representing the order of the ARIMA model, which consists of (p, d, q) parameters where 'p' is the number of AR terms, 'd' is the degree of differencing, and 'q' is the number of MA terms.

### Pseudo Code Algorithm

```
Function Parameter_calculation with parameter: data
    Run auto_arima on 'data' with tracing enabled to find optimal parameters
    Plot the ACF of 'data' using the global 'lag' variable
    Save the ACF plot to the 'Output_address' directory with the filename "ACF.jpg"
    Plot the PACF of 'data' using the global 'lag' variable
    Save the PACF plot to the 'Output_address' directory with the filename "PACF.jpg"
    Extract the order (p, d, q) of the ARIMA model from the findings of auto_arima
    Return the order of the ARIMA model
EndFunction
```

### Flow of the Program for `Parameter_calculation()`

1. Execute the `auto_arima` function on the input `data` to automatically determine the best-fitting ARIMA model parameters while printing the trace of the fitting process.
2. Plot the ACF for the given `data` up to the number of lags specified by `lag`.
3. Save the ACF plot to the specified `Output_address` directory with the appropriate filename.
4. Plot the PACF for the given `data` up to the number of lags specified by `lag`.
5. Save the PACF plot to the specified `Output_address` directory with the appropriate filename.
6. Retrieve the order of the ARIMA model (p, d, q) from the results of the `auto_arima` function.
7. Return the ARIMA model order for use in subsequent model fitting.

### Function Definition and Working

#### `ARIMA_Model(train, len_test, ord)`

**Purpose:**  
The `ARIMA_Model` function fits an ARIMA model to the training data and uses it to make predictions. The primary use in this context is to forecast the potential errors from an LSTM model, which can then be used for error correction in the LSTM's predictions.

**Input:**
- `train`: A pandas Series or numpy array containing the training set data used to fit the ARIMA model.
- `len_test`: An integer representing the length of the test dataset, which dictates how many future steps to predict.
- `ord`: A tuple indicating the order of the ARIMA model, typically obtained from the `Parameter_calculation` function, which consists of (p, d, q) parameters.

**Processing Elements:**
1. **ARIMA from statsmodels:** A class that represents an ARIMA model, used here for time series forecasting.
2. **Fitting the Model:** The ARIMA model is fitted to the training data using the provided order parameters.
3. **Predictions:** The model is used to make predictions for the specified future time steps.

**Output:**
- `model`: The fitted ARIMA model object.
- `predictions`: The forecasts from the model starting from the end of the training set to the length of the test set.
- `full_predictions`: The full set of in-sample predictions for the training data.

### Pseudo Code Algorithm

```
Function ARIMA_Model with parameters: train, len_test, ord
    Initialize an ARIMA model with 'train' data and 'ord' order
    Fit the ARIMA model to the 'train' data
    Make predictions from the end of 'train' data up to the length of the test set plus one
    Make full in-sample predictions for the 'train' data
    Return the fitted model, out-of-sample predictions, and in-sample predictions
EndFunction
```

### Flow of the Program for `ARIMA_Model()`

1. Instantiate an ARIMA model with the training data `train` and the order parameters `ord`.
2. Fit the model to the training data using the `fit()` method.
3. Use the `predict` method of the fitted model to forecast future values for a range starting at the end of the training set and extending `len_test` steps into the future.
4. Also, generate a full set of in-sample predictions for the training data, which covers the entire range of the training set.
5. Return the fitted ARIMA model, the out-of-sample predictions for error correction, and the in-sample predictions for evaluation purposes.

### Function Definition and Working

#### `Final_Predictions(predictions_errors, predictions)`

**Purpose:**  
The `Final_Predictions` function calculates the final forecasted values by adjusting the LSTM model predictions with the ARIMA model-predicted errors. This technique is often used in hybrid models to correct predictions from one model using insights from another.

**Input:**
- `predictions_errors`: A list or pandas Series containing the errors between the actual values and the LSTM model's predictions, as forecasted by the ARIMA model.
- `predictions`: A list or pandas Series containing the LSTM model's predictions.

**Processing Elements:**
1. **List Iteration:** A loop that runs through the number of `days` (a globally set variable), combining the predictions from the LSTM model and the errors predicted by the ARIMA model.

**Output:**
- `final_values`: A list of the corrected predictions after accounting for the ARIMA-predicted errors.

### Pseudo Code Algorithm

```
Function Final_Predictions with parameters: predictions_errors, predictions
    Initialize an empty list 'final_values'
    Loop over the range of 'days' (global variable):
        Calculate the final value by adding the prediction error to the LSTM prediction at each index
        Append the final value to 'final_values'
    Return 'final_values'
EndFunction
```

### Flow of the Program for `Final_Predictions()`

1. Start by creating an empty list `final_values` to store the adjusted predictions.
2. Loop through a range of indices defined by the global variable `days`, which determines how many final predictions to calculate.
3. At each iteration, add the corresponding prediction error from `predictions_errors` to the LSTM prediction from `predictions` and append the result to `final_values`.
4. After the loop completes, return `final_values`, which contains the final adjusted predictions.


### Function Definition and Working

#### `main()`

**Purpose:**  
The `main` function orchestrates the entire process of loading data, preparing it, training the LSTM model, making predictions, evaluating errors, and generating various plots and outputs. It serves as the entry point for running the time series forecasting program.

**Input:**  
There are no direct inputs to the `main` function as it stands alone. It relies on global variables and the functions it calls to operate on the data.

**Processing Elements:**
1. **Data loading and plotting functions:** `data_loader`, `plot_raw_data`
2. **Data partitioning function:** `data_allocation`
3. **Model training and prediction functions:** `LSTM`, `Error_Evaluation`, `Parameter_calculation`, `ARIMA_Model`, `Final_Predictions`
4. **Accuracy calculation functions:** `calculate_accuracy`
5. **Plotting accuracy and errors:** `plot_train_test`, `plot_predictions`, `plot_prediction_errors`, `plot_final_predictions`, `plot_accuracy`, `plot_arima_accuracy`
6. **File writing:** Outputs model summaries and predictions to a text file.

**Output:**  
The `main` function does not return any value. Its outputs are:
- Plots saved as images in the specified output directory.
- Console prints of model summaries and accuracy metrics.
- A text file saved with detailed model information and predictions.

### Pseudo Code Algorithm

```
Function main
    Load data using data_loader function
    Plot raw data using plot_raw_data function
    Partition data into training and testing sets using data_allocation function
    Plot training and testing data using plot_train_test function
    
    Start timing the LSTM model process
    Train LSTM model using LSTM function
    Plot LSTM predictions using plot_predictions function
    Make new predictions using the trained LSTM model
    
    Evaluate errors in LSTM predictions using Error_Evaluation function
    Plot prediction errors using plot_prediction_errors function
    Calculate accuracy of LSTM predictions using calculate_accuracy function
    Plot LSTM accuracy using plot_accuracy function
    
    Determine ARIMA model parameters using Parameter_calculation function
    Fit ARIMA model and make predictions on errors using ARIMA_Model function
    Calculate ARIMA model accuracy and plot it using plot_arima_accuracy function
    
    Calculate final predictions by combining LSTM predictions and ARIMA predicted errors using Final_Predictions function
    Plot final predictions using plot_final_predictions function
    
    Write LSTM and ARIMA model details, predictions, and accuracies to an output text file
    Print the time taken for the entire process
EndFunction

Call main function if the script is the main program
```

### Flow of the Program for `main()`

1. Call `data_loader` to load the dataset.
2. Call `plot_raw_data` to visualize the raw dataset.
3. Call `data_allocation` to split the data into training and testing sets.
4. Call `plot_train_test` to visualize the training and testing datasets.
5. Train the LSTM model by calling `LSTM` and plot its predictions.
6. Generate predictions for the test set using the trained LSTM model.
7. Evaluate prediction errors with `Error_Evaluation` and visualize them.
8. Calculate and print the LSTM model's accuracy, plotting the results.
9. Determine the best ARIMA model parameters and fit the ARIMA model to predict errors.
10. Plot ARIMA model accuracy.
11. Combine LSTM predictions with ARIMA-predicted errors using `Final_Predictions` and visualize the final predictions.
12. Write all relevant outputs, including model summaries, accuracies, and predictions, to a text file.
13. Print the total time taken for the process.
14. Execute the `main` function if the script is run as the main program.


## DataSet 1 : 

## Plots and Outputs :

### RAW DATA : 

<br>

![plot](../programs/Output_Dataset_01/Raw%20Time%20Series%20Data.jpg)

### Train Data and Test Data 

<br>

![plot](../programs/Output_Dataset_01/Train%20and%20Test%20Data.jpg)

### LSTM Predicted Train Data Vs Actual Train Data Value

<br>

![plot](../programs/Output_Dataset_01/LSTM%20PREDICTIONS%20VS%20ACTUAL%20Values%20For%20TRAIN%20Data%20Set.jpg)

### ERROR Evaluation on Train Data 

<br>

![plot](../programs/Output_Dataset_01/Prediction%20Errors%20over%20Time.jpg)


### ARIMA Model


### ACF :

![plot](../programs/Output_Dataset_01/ACF.jpg)

### PACF : 

![plot](../programs/Output_Dataset_01/PACF.jpg)

### Arima Model Accuracy 

![plot](../programs/Output_Dataset_01/Model%20Accuracy%20Metrics.jpg)

### LSTM Predictions (over test data)

<br>

![plot](../programs/Output_Dataset_01/LSTM%20Predictions%20VS%20Actual%20Values.jpg)


### FINAL Predictions (ARIMA + LSTM)

<br>

![plot](../programs/Output_Dataset_01/Final%20Predictions%20with%20Error%20Correction.jpg)


## Analysis of Predictive Performance and Accuracy of the Individual & Hybrid Model :

### LSTM Model Configuration

| Parameter        | Value |
|------------------|-------|
| Lags Used        | 12    |
| Epochs           | 15    |
| Learning Rate    | 0.001 |
| Batch Size       | 32    |
| Number of Nodes  | 32    |
| Lag for NN       | 12    |

### Analysis of Configuration Values:

The configuration of the LSTM model as described indicates a moderate complexity with enough capacity to capture trends in time-series data. A lag value of 12 suggests that the model looks at the last 12 time points to make a prediction. The number of epochs is set to 15, which is a reasonable number for training without overfitting, assuming a decent-sized dataset.

A learning rate of 0.001 is standard for many LSTM models, as it's slow enough to converge smoothly but not too slow to stall the training process. The batch size of 32 is a common choice that balances the speed of computation with the stability of the gradient descent optimization.

The number of nodes, 32, indicates that each LSTM cell (and subsequent dense layers) has 32 units. This is sufficient for capturing the complexities in the data without being overly prone to overfitting. Given these parameters, the model is likely to perform well on time series forecasting tasks with a moderate amount of data points.



### LSTM Model Full Predictions for Train Data (First 100 Points)

| Index | Actual Data Point | Predicted Data Point |
|-------|-------------------|----------------------|
| 0     | 10.441108         | 9.976713            |
| 1     | 10.359913         | 9.616003            |
| ...   | ...               | ...                 |
| 98    | 12.95306          | 12.643594           |
| 99    | 12.696795         | 13.044828           |

(Note: The table above is truncated to conserve space; full data should be tabulated similarly.)

### Analysis of Predicted Values:

The LSTM model's predictions show a degree of variance when compared to the actual data points. In many cases, the model seems to underestimate the actual values, although there are instances where it predicts higher values. This variance suggests that while the model has learned the underlying pattern of the dataset, there is still room for improvement in terms of prediction accuracy.

The differences between the actual and predicted values can be used to fine-tune the model parameters, such as the number of epochs, learning rate, and the architecture itself, to achieve more accurate forecasts. Moreover, the discrepancies can also guide the exploration of additional features or alternative modeling techniques to capture the data's behavior more effectively.

For a comprehensive evaluation, it would be useful to consider other metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), or Mean Absolute Percentage Error (MAPE), which can provide additional insight into the model's performance across the dataset.


### LSTM Model Accuracy Metrics

| Metric                         | Value                          |
|--------------------------------|--------------------------------|
| Mean Squared Error (MSE)       | 3044.490484928724              |
| Root Mean Squared Error (RMSE) | 55.176901733684936             |
| Mean Absolute Error (MAE)      | 52.61303724687495              |

### Analysis of Accuracy Metrics:

The provided metrics indicate the model's performance in terms of error magnitudes. The Mean Squared Error (MSE) is relatively high, which could suggest significant variance in the model's predictions. Since MSE is sensitive to outliers, this could be influenced by a few large errors.

The Root Mean Squared Error (RMSE) provides a more interpretable measure as it is in the same units as the data. An RMSE of approximately 55.18 means that the model’s predictions are, on average, about 55 units away from the actual values. This can be considered a moderate error depending on the scale and variance of the data.

The Mean Absolute Error (MAE) of about 52.61 is the average magnitude of errors in the predictions. Like RMSE, the MAE suggests that predictions may vary moderately from the actual values. However, it does not square the errors, so it is not as sensitive to large errors as MSE and RMSE.

Overall, these metrics suggest that while the model is capable of predicting the general trend, there is a notable average error per prediction. Depending on the specific application and the range of the data, this level of error may or may not be acceptable, and further model refinement might be necessary.

Here's the provided data for the LSTM model's 10-day predictions in table format:

### LSTM Model 10-Day Predictions

| Day | Actual Value | Predicted Value     |
|-----|--------------|---------------------|
| 1   | 2312.5       | 2335.640625         |
| 2   | 2287.899902  | 2368.354736328125   |
| 3   | 2297.399902  | 2355.52783203125    |
| 4   | 2320.199951  | 2360.1640625        |
| 5   | 2319.699951  | 2374.385986328125   |
| 6   | 2339.0       | 2375.121337890625   |
| 7   | 2323.800049  | 2385.907470703125   |
| 8   | 2335.899902  | 2378.192138671875   |
| 9   | 2310.550049  | 2384.339111328125   |
| 10  | 2314.899902  | 2370.3466796875     |

###  Analysis of Prediction Values:

The predictions made by the LSTM model are generally above the actual values, indicating a consistent overestimation across the 10-day period. This pattern suggests that the model may be capturing a trend of increasing values, or it may be biased towards higher predictions due to the underlying data it was trained on.

The difference between the predicted and actual values varies, with the model's predictions ranging approximately between 20 to 60 units higher than the actual values. The consistent overestimation across all days could point to specific features in the data that the model is perhaps weighting too heavily, or it might indicate the model's sensitivity to certain trends or noise within the training dataset.

Considering the scale of actual values (all being above 2000), the absolute errors are relatively small, which may be acceptable in some applications. However, for others, especially those requiring precise forecasting

Here's the ARIMA model summary presented in a table format:

### ARIMA Model Summary

| Parameter      | Value                |
|----------------|----------------------|
| Dep. Variable  | y                    |
| No. Observations| 6981               |
| Model          | ARIMA(1, 0, 1)       |
| Log Likelihood | -30220.938           |
| AIC            | 60449.876            |
| BIC            | 60477.280            |
| HQIC           | 60459.320            |
| Covariance Type| opg                  |

### Model Coefficients

| Coefficient | Value  | Std Err  | z     | P>\|z\| | [0.025 | 0.975] |
|-------------|--------|----------|-------|---------|--------|--------|
| const       | -0.0639| 2.436    | -0.026| 0.979   | -4.837 | 4.710  |
| ar.L1       | 0.9588 | 0.001    | 691.112| 0.000  | 0.956  | 0.962  |
| ma.L1       | -0.5480| 0.004    | -137.501| 0.000 | -0.556 | -0.540 |
| sigma2      | 336.9694| 1.757   | 191.750| 0.000 | 333.525| 340.414|

### Other Statistical Measures

| Measure           | Value        |
|-------------------|--------------|
| Ljung-Box (Q)     | 0.00         |
| Prob(Q)           | 1.00         |
| Jarque-Bera (JB)  | 113053.71    |
| Prob(JB)          | 0.00         |
| Heteroskedasticity(H) | 503.23   |
| Skew              | 0.07         |
| Kurtosis          | 22.71        |

### Analysis of ARIMA Model Values:

The ARIMA(1, 0, 1) model indicates a single autoregressive term and a single moving average term, which suggests a relatively simple model. The coefficients for both `ar.L1` and `ma.L1` are statistically significant (p<0.001), indicating a good fit for the data.

The large value of the Jarque-Bera test statistic and the corresponding probability close to zero suggest that the residuals of the model do not follow a normal distribution, exhibiting high kurtosis and slight skewness.

The heteroskedasticity measure is very high, which may indicate that the error variance changes over time. This could suggest the presence of volatility clustering, which the model may not fully capture.

Despite these indications of non-normality and heteroskedasticity, the model's AIC and BIC suggest it has a reasonable fit to the data. However, due to the identified issues, there may be room for further model improvement or the need for a different modeling approach to better account for the data's characteristics.

### ARIMA Model 10 Prediction Errors

| Prediction Index | Error Value           |
|------------------|-----------------------|
| 0                | -63.692812476496954   |
| 1                | -61.07029188965002    |
| 2                | -58.55586074208999    |
| 3                | -56.14506403470495    |
| 4                | -53.83363038499918    |
| 5                | -51.61746445917654    |
| 6                | -49.49263971614206    |
| 7                | -47.45539145056572    |
| 8                | -45.50211012268237    |
| 9                | -43.629334963009626   |
| 10               | -41.833747840652855   |

### Analysis of Prediction Error Values:

The table presents the predicted errors from the ARIMA model for a sequence of forecasted points. The negative values across all predictions indicate that the ARIMA model consistently forecasts lower than the actual values (underestimation).

The magnitude of the errors seems to decrease with each subsequent prediction, indicating the model's increasing accuracy or the convergence of the model's predictions to the actual data points over time. However, the presence of negative values also suggests that if these errors are to be used to adjust another model's forecasts (such as an LSTM model), they would need to be added to increase the forecasted values, potentially correcting for an underlying underestimation tendency.

It's important to note that while the trend of decreasing error magnitude is a positive sign, the actual values of the errors are relatively large. This may point to the need for further model optimization or the exploration of additional data features that could improve the accuracy of the ARIMA model's predictions.

### Comparison of LSTM and Final Predictions with Actual Values Over 10 Days

| Day | Actual Value | LSTM Predicted Value | Final Prediction (LSTM + ARIMA) |
|-----|--------------|----------------------|---------------------------------|
| 0   | 2312.5       | 2335.640625         | 2271.947812523503              |
| 1   | 2287.899902  | 2368.354736328125   | 2307.284444438475              |
| 2   | 2297.399902  | 2355.52783203125    | 2296.97197128916               |
| 3   | 2320.199951  | 2360.1640625        | 2304.018998465295              |
| 4   | 2319.699951  | 2374.385986328125   | 2320.552355943126              |
| 5   | 2339.0       | 2375.121337890625   | 2323.5038734314485             |
| 6   | 2323.800049  | 2385.907470703125   | 2336.414830986983              |
| 7   | 2335.899902  | 2378.192138671875   | 2330.736747221309              |
| 8   | 2310.550049  | 2384.339111328125   | 2338.8370012054424             |
| 9   | 2314.899902  | 2370.3466796875     | 2326.7173447244904             |

### Forecast of Next Data Point

| Forecast for Next Data Point | Value             |
|------------------------------|-------------------|
| Prediction                   | 2329.981681846847 |

### Analysis of Prediction Values:

The LSTM model's predictions are consistently higher than the actual values, indicating a systematic overestimation trend. After incorporating the ARIMA model's error predictions, the final adjusted predictions are closer to the actual values, reducing the prediction error and improving the accuracy of the forecast.

The ARIMA model corrections have effectively adjusted the LSTM predictions towards the actual values, demonstrating the utility of combining models to correct for biases in time series forecasting. The systematic adjustments suggest that the ARIMA model's error predictions are valuable for fine-tuning the LSTM outputs.

The forecast for the next data point, 2329.981681846847, is an estimate based on the combined predictive behavior of the LSTM and ARIMA models, providing a corrected prediction that reflects both models' insights.

The time taken for model training and predictions, 25.10 seconds, indicates a relatively efficient computational performance, which is beneficial for practical applications requiring timely forecasts.

---


## DataSet 2 : 

## Plots and Outputs :

### RAW DATA : 

<br>

![plot](../programs/Output_Dataset_02/Raw%20Time%20Series%20Data.jpg)

### Train Data and Test Data 

<br>

![plot](../programs/Output_Dataset_02/Train%20and%20Test%20Data.jpg)

### LSTM Predicted Train Data Vs Actual Train Data Value

<br>

![plot](../programs/Output_Dataset_02/LSTM%20PREDICTIONS%20VS%20ACTUAL%20Values%20For%20TRAIN%20Data%20Set.jpg)

### ERROR Evaluation on Train Data 

<br>

![plot](../programs/Output_Dataset_02/Prediction%20Errors%20over%20Time.jpg)


### ARIMA Model


### ACF :

![plot](../programs/Output_Dataset_02/ACF.jpg)

### PACF : 

![plot](../programs/Output_Dataset_02/PACF.jpg)

### Arima Model Accuracy 

![plot](../programs/Output_Dataset_02/Model%20Accuracy%20Metrics.jpg)

### LSTM Predictions (over test data)

<br>

![plot](../programs/Output_Dataset_02/LSTM%20Predictions%20VS%20Actual%20Values.jpg)


### FINAL Predictions (ARIMA + LSTM)

<br>

![plot](../programs/Output_Dataset_02/Final%20Predictions%20with%20Error%20Correction.jpg)


## Analysis of Predictive Performance and Accuracy of the Individual & Hybrid Model :

### LSTM Model Configuration

| Parameter        | Value |
|------------------|-------|
| Lags Used        | 12    |
| Epochs           | 15    |
| Learning Rate    | 0.001 |
| Batch Size       | 32    |
| Number of Nodes  | 32    |
| Lag for NN       | 12    |

### Analysis of Configuration Values:

The configuration of the LSTM model as described indicates a moderate complexity with enough capacity to capture trends in time-series data. A lag value of 12 suggests that the model looks at the last 12 time points to make a prediction. The number of epochs is set to 15, which is a reasonable number for training without overfitting, assuming a decent-sized dataset.

A learning rate of 0.001 is standard for many LSTM models, as it's slow enough to converge smoothly but not too slow to stall the training process. The batch size of 32 is a common choice that balances the speed of computation with the stability of the gradient descent optimization.

The number of nodes, 32, indicates that each LSTM cell (and subsequent dense layers) has 32 units. This is sufficient for capturing the complexities in the data without being overly prone to overfitting. Given these parameters, the model is likely to perform well on time series forecasting tasks with a moderate amount of data points.


### LSTM Full Predictions vs. Actual Data Points (First 100 Points)

| Index | Actual Data Point | Predicted Data Point |
|-------|-------------------|----------------------|
| 0     | 16.508291         | 17.38346290588379    |
| 1     | 16.235182         | 17.361835479736328   |
| ...   | ...               | ...                  |
| 98    | 18.200382         | 18.093708038330078   |
| 99    | 17.847122         | 18.31888771057129    |

(Note: Table truncated for brevity, full data spans from index 0 to 99.)

###  Analysis of Prediction Values:

The LSTM model's predictions are presented alongside the actual data points for comparison. A quick overview of the table suggests the model is generally predicting higher values than the actual data, indicating a tendency of the model to overestimate. 

The consistency in overestimation could be due to various factors such as model overfitting, a lack of sufficient training data diversity, or the model's sensitivity to the particular features of the data set. It might also indicate that the model's parameters, such as the number of layers, nodes, or the learning rate, need adjustment.

A deeper analysis could involve examining the residuals (the differences between actual and predicted values) to identify any patterns or biases in the model's predictions. By understanding these patterns, further tuning and improvements can be made to the model to enhance its predictive performance.

For the sake of a concise representation, the analysis is general, and a detailed statistical analysis would require a thorough examination of the entire set of predictions alongside comprehensive diagnostic checks of the model's assumptions and performance metrics.

### LSTM Model Performance Metrics

| Metric                  | Value                     |
|-------------------------|---------------------------|
| Mean Squared Error (MSE)| 1404.5218992265777        |
| Root Mean Squared Error (RMSE)| 37.47695157328805  |
| Mean Absolute Error (MAE)| 36.46679693710937        |

(Note: The history object is not included in the table as it is a reference to the object in memory.)

### Short Analysis of Performance Metrics:

The Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are both measures of the average squared difference between the predicted values and the actual values, with RMSE being particularly sensitive to larger errors due to the squaring of the residuals. The relatively high values of MSE and RMSE indicate that on average, the model's predictions deviate from the actual values by a margin that could be considered significant in many contexts, suggesting room for improvement in the model's accuracy.

The Mean Absolute Error (MAE) measures the average absolute difference between predicted and actual values, providing a more intuitive sense of the average error magnitude. A MAE of approximately 36.47 suggests that, on average, the model's predictions are about 36.47 units away from the actual values.

Together, these metrics suggest that the model has a certain level of predictive capability but also highlight the potential for model refinement to reduce predictive errors and improve accuracy. Potential strategies for improvement might include hyperparameter tuning, model complexity adjustments, or additional feature engineering.

### LSTM Model 10-Day Predictions

| Day | Actual Value | Predicted Value   |
|-----|--------------|-------------------|
| 1   | 898.950012   | 860.52197265625   |
| 2   | 883.099976   | 861.7955932617188 |
| 3   | 890.0        | 856.0772705078125 |
| 4   | 880.349976   | 858.2377319335938 |
| 5   | 891.099976   | 854.6444091796875 |
| 6   | 893.549988   | 858.5345458984375 |
| 7   | 900.5        | 859.7453002929688 |
| 8   | 903.150024   | 862.2251586914062 |
| 9   | 911.25       | 863.2906494140625 |
| 10  | 913.700012   | 865.9093627929688 |

###  Analysis:

The table shows the LSTM model's predicted values against the actual values over a span of ten days. Observing the differences between the actual and predicted values, we can see that the model tends to underpredict the actual values consistently across all ten days.

The model's predictions are trailing behind the actual stock prices by a certain margin, which suggests that the model may be conservative in its forecasting or may not be capturing some trend or pattern in the data effectively. This could potentially be due to various factors such as the model's sensitivity to recent data points, the selection of input features, or the complexity of the model itself.

To improve the model's performance, one might consider revisiting the model architecture, tuning hyperparameters, or incorporating additional data that could help the model better understand the underlying trends in the stock prices.


### ARIMA Model Summary for SARIMAX Results

| Parameter | Coefficient | Std. Err. | z    | P>\|z\| | [0.025 | 0.975] |
|-----------|-------------|-----------|------|---------|--------|--------|
| ar.L1     | -1.2284     | 0.005     | -232.093 | 0.000   | -1.239 | -1.218 |
| ar.L2     | -0.2509     | 0.005     | -53.372  | 0.000   | -0.260 | -0.242 |
| ma.L1     | 0.0823      | 0.004     | 21.912   | 0.000   | 0.075  | 0.090  |
| ma.L2     | -0.8920     | 0.004     | -240.693 | 0.000   | -0.899 | -0.885 |
| sigma2    | 21.0358     | 0.110     | 190.902  | 0.000   | 20.820 | 21.252 |

### Model Statistics

| Statistic        | Value      |
|------------------|------------|
| Log Likelihood   | -20521.972 |
| AIC              | 41053.944  |
| BIC              | 41088.194  |
| HQIC             | 41065.748  |
| No. Observations | 6976       |

### Tests and Diagnostics

| Test                | Value       |
|---------------------|-------------|
| Ljung-Box (Q)       | 0.00        |
| Jarque-Bera (JB)    | 106465.57   |
| Prob(Q)             | 0.97        |
| Prob(JB)            | 0.00        |
| Heteroskedasticity (H) | 175.38   |
| Skew               | 0.23        |
| Kurtosis           | 22.13       |

###  Analysis:

The ARIMA model presented utilizes a second-order autoregressive process (ar.L1 and ar.L2) and a second-order moving average process (ma.L1 and ma.L2). The coefficients for the AR terms are negative, indicating a possible overcompensation for previous values. The moving average parameters show that the model significantly weights the recent errors in prediction.

The model's AIC and BIC values are high, suggesting that the model is complex, which is typical for time series data with many observations. A lower AIC or BIC might suggest a better model fit.

The Ljung-Box test shows a p-value close to 1, indicating that the residuals are independently distributed (no autocorrelation). The Jarque-Bera test indicates that the residuals do not follow a normal distribution, given the very high test statistic and a p-value of 0.00. This could suggest that the model might not be fully capturing the underlying process.

Heteroskedasticity is present, given the high value, which could indicate that the variance of the residuals is not constant over time.

The skewness is close to 0, suggesting the residuals are symmetrically distributed, while the kurtosis indicates a leptokurtic distribution, meaning there are outliers or extreme values more than a normal distribution would expect.


### ARIMA Model 10 Predictions Table

| Prediction No. | Predicted Value |
|----------------|-----------------|
| 1              | 30.9800         |
| 2              | 31.2458         |
| 3              | 31.7955         |
| 4              | 31.0536         |
| 5              | 31.8271         |
| 6              | 31.0630         |
| 7              | 31.8076         |
| 8              | 31.0846         |
| 9              | 31.7859         |
| 10             | 31.1058         |
| 11             | 31.7653         |

###  Analysis:

The values presented in the table are the predictions from the ARIMA model for ten different points in time. These predictions are likely generated from a test dataset where the ARIMA model has been applied to forecast future values based on past data.

The predicted values seem to be clustered around the 31-unit mark, indicating that the model forecasts a relatively stable trend around this value. The slight fluctuations between the predictions suggest some variability, but without further context on the scale of the data and the nature of what's being predicted, it's challenging to draw more detailed conclusions.

Overall, these predictions could be used to make informed decisions in the context they are applied to, such as stock prices, weather forecasting, or any other time series data. However, the accuracy and utility of these predictions would depend on how well the ARIMA model fits the historical data and captures the underlying patterns.

### Difference Between the LSTM and Final Predictions Table

| Day | Actual Value | LSTM Predicted Value | Final Prediction (LSTM + ARIMA) |
|-----|--------------|----------------------|---------------------------------|
| 1   | 898.95       | 860.52               | 891.50                          |
| 2   | 883.10       | 861.80               | 893.04                          |
| 3   | 890.00       | 856.08               | 887.87                          |
| 4   | 880.35       | 858.24               | 889.29                          |
| 5   | 891.10       | 854.64               | 886.47                          |
| 6   | 893.55       | 858.53               | 889.60                          |
| 7   | 900.50       | 859.75               | 891.55                          |
| 8   | 903.15       | 862.23               | 893.31                          |
| 9   | 911.25       | 863.29               | 895.08                          |
| 10  | 913.70       | 865.91               | 897.02                          |

### Analysis:

This table shows a comparison between the actual values, the values predicted by an LSTM model, and the final predictions which seem to be a corrected version possibly by adding the error predicted by an ARIMA model.

The LSTM model's predictions are consistently lower than the actual values, indicating a systematic underestimation. The final predictions, which incorporate both LSTM and ARIMA models, are closer to the actual values, suggesting that the ARIMA model's error correction is beneficial in improving prediction accuracy.

The time taken for model training and predictions indicates the computational efficiency, with just under 51 seconds showing a relatively swift processing time for this task.

The forecast value for the next data point suggests the model's prediction for the immediate future, assuming no actual value is available for comparison. This figure provides an estimated outcome based on the trained model's understanding of the data trends.
## Methods

### Learning Algorithm Overview

The aim of the learning algorithm was to:

- Generate $X\in\mathbb{R}^2$ 10-Gaussian Mixture Models for 20 simulations
- Evaluate performance of the algorithms mentioned above
- Find optimal $k$-value for k-Nearest Neighbors via 10-Fold Cross Validation.

#### Parameters

This learning algorithm was parameterized as follows:

- $n\text{-simulations}$: = 20
- $n\text{-gaussians}$: = 10
- $\sigma\text{-centers}$: = 2
- $\sigma\text{-data}$: = $\sqrt{\frac{4}{5}}$
- $k\text{-folds}$: = 10
- $n\text{-training}$: 100 (per class)
- $n\text{-test}$: 5000 (per class)

#### Process

Upon instantiation,  a data generation step creates the centers for the 10 Gaussian Mixture Models. During each of the following 20 simulations, data were generated, and the algorithms were trained and evaluated.  The overall methodology is summarized in the following pseudocode.

1. Initialize parameters
2. Generate centers
3. For i=0; i<$n\text{-simulations}$; ++i
   1. Generates data
   2. Train and evaluate Linear Regression
   3. Train and evaluate Quadratic Regression
   4. Train and evaluate Bayes' Rule 
   5. Train and evaluate k-Nearest Neighbors
   6. Train and evaluate Logistic Regression
4. Report results

#### Output

Accuracy and Area Under the ROC Curve (AUC) were the performance measures used to evaluate performance. As such, the algorithm produces:

- Data: The training data for two randomly selected simulations are rendered via scatterplot
- Performance: 
  - Line plots showing the training and test scores by simulation
  - Box plots evincing the distribution of training and test scores
  - Descriptive statistics for training and test scores
- k-Nearest Neighbors
  - The optimal k-value is computed via 10-fold cross-validation and presented
  - Descriptive statistics for the k-values 

### Environment

All experiments were performed on:

| Processor:        | Intel® Xeon® CPU X5650 @ 2.67Ghz (Dual   Processors) |
| ----------------- | ---------------------------------------------------- |
| Installed RAM:    | 128 GB                                               |
| Operating System: | Windows 10 Professional Build 19041.804              |
| System Type:      | 64-bit operating system, x64-based processor         |

The software was implemented in Python 3.9.

#### Reproducibility

A seed was set at instantiation at 6998, the last four digits of the author's University ID. During the data generation process, the seed was incremented by 1 at each iteration to ensure that the pseudorandom process produced different data for each simulation.

The learning algorithm is comprised of the following modules:

| Module        | Class                | Description                                                  |
| ------------- | -------------------- | ------------------------------------------------------------ |
| Data          | DataGen              | Generates Guassian centers and the data                      |
|               | Kfold                | Generates the data for 10 Fold Cross-Validation              |
| Model         | Linear Regression    | Trains and scores linear regression model                    |
|               | Quadratic Regression | Trains and scores quadratic regression model                 |
|               | Bayes' Rule          | Trains and scores Bayes' Rule model                          |
|               | k-Nearest Neighbors  | Trains and scores k-Nearest Neighbors Model                  |
|               | Logistic Regression  | Trains and scores logistic regression model                  |
| Visualization | Visualizer           | Base class for Visualizer classes                            |
|               | DataVisualizer       | Renders data plots                                           |
|               | ScoreVisualizer      | Presents plots and descriptive statistics for training and test scores |
|               | KNNVisualizer        | Plots histogram of k-values as well as the distribution of best k's |
| Learner       | Simulation           | Performs a single simulation                                 |
|               | Learner              | Learning algorithm                                           |

Next, we will examine each module in detail.

### Data Generation

#### DataGen

This class has two main methods. The generate_centers method produces the 10 centers for each class at instantiation. The generate_data method is called once per simulation and produces training and test data. Note that the seed is incremented by 1 each time the method is called in order to generate different data each simulation.

[code]

#### KFold

This method produces the data used during k-Nearest Neighbors 10-Fold Cross-Validation. It shuffles and stratifies the data to ensure that the classes are balanced.

[code]

### Models

The five classes in this module are responsible for fitting, predicting, and scoring the models. In addition, the best k-value is computed using 10-fold cross validation. To address the issue of **non-uniqueness of optimal k-values**, the best-k value was selected using the so-called **one standard error** rule. The largest k that produced a training error within one standard error of the minimum training error was selected.

[code]

### Visualizations

The three classes, DataVisualizer, ScoreVisualizer, and kNNVisualizer, provide the plots and the tabular descriptive statistics.

[code]

### Learner

This module contains two classes, the Simulator class and the Learner class. The former is the driver for each simulation. It generates the data, trains the models, and performs the evaluation. The latter initiates the process 

[code]

### Main

Finally, the main module.

## Results




# DeepLearningProject2
### UNM CS429
# Authors
### Benjamin Liu, Danyelle Loffredo, and Jared Bock


# About

* This project uses the Naive Bayes Algorithm and Gradient Descent with Logisitc Regression to classify text documents into one of 20 newscast topics. This is done by calculating the probabilities that a word appears in a class based on training data and using these probabilities to predict the probability that the sample belongs to a certain class. The Logistic Regression algorithm uses Gradient Descent to minimize our cost function and find an optimal weights, W, that can be then used to in the equation exp(WX<sup>T</sup>) which will then be used to find P(Y|W,X)

# Installation

To install this project, you can clone the git repo to your local machine or, in GitHub, under the ```Code``` tab, click on the Green Code button, and the choose ```Download Zip```. 

![](Resources/codebutton.png)

# Usage 
For running Naive Bayes from the command line in a Linux terminal:

* cd into the directory where NaiveBayes.py, "training.csv", "testing.csv" are located (these should all be the same directory with the exact names listed).

* Once in the correct directory, simply type 
```python NaiveBayes.py``` into the terminal.

* For Windows, the process is the same, however to run, you will need to enter ```py NaiveBayes.py``` and follow the prompts outlined as above.

* A second option for running the code on a Linux/UNIX system is, once located in the correct directory, type ``` make run``` into the terminal.

For running Logistic regression from the command line in a Linux terminal:

* cd into the directory where LogReg.py, "training.csv", "testing.csv" are located (these should all be the same directory with the exact names listed).

* Once in the correct directory, simply type 
```python LogReg.py``` into the terminal.

* For Windows, the process is the same, however to run, you will need to enter ```py LogReg.py``` and follow the prompts outlined as above.

* A second option for running the code on a Linux/UNIX system is, once located in the correct directory, type ``` make run``` into the terminal.

# Our environment
```bash
python --version
3.8.3
pandas.__version__
1.0.5
scipy.__version__
1.8.0
matplotlib.__version__
3.5.1


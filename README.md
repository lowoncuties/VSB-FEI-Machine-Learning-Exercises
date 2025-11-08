#  Machine Learning
## The course is held at FEI, VSB-TU Ostrava

Course information may be found [here](https://homel.vsb.cz/~pla06/subject_ml.html).

Feel free to contact me (<lukas.jochymek.st@vsb.cz>) if you have any questions or want to discuss any topic from the course 😊

All authorship is mentioned where possible.

**Thanks to Radek Svoboda for inspiration and memes, that will be used in these materials.**

## 📚 Study materials

Materials that should help you with the basic concepts from the mathematics and statistics needed by this course.

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Fundamentals-of-Machine-Learning-Exercises/blob/master/statistics_explained.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Fundamentals-of-Machine-Learning-Exercises/blob/master/statistics_explained.ipynb)


Any PR is welcome, whether you find a typo or you have better explanation

# 📊 Exercises
## Exercise 0
The aim of the exercise is to get an overview of the course, get familiar with the jupyter notebooks and be able to setup a Python Virtual Enviroment (`venv`)

## Exercise 1
The task of the exercise will be to implement the Apriori/ProjectedEnumerationTree algorithm for pattern mining. Individual tasks for the exercise:

1. Generate all combinations without repetition of length 3 from 6 possible ones.
2. On one of the test files (chess, connect), generate numerous patterns and calculate Support.
3. From the generated frequent patterns, write down the rules and their Confidence.

<details>
  <summary><strong>Test results</strong></summary>

  Selected results for dataset Test, min_support>=0.25, min_confidence>=0.5.
  Number of frequent patterns meeting min_support:

      1 element: 5
      2 elements: 5

  Patterns exceeding min_confidence:

      3 -> 1 (conf=0.6)
      5 -> 1 (conf=0.71)
      1 -> 5 (conf=0.71)
      2 -> 5 (conf=0.75)
      3 -> 5 (conf=0.6)
      4 -> 5 (conf=0.75)

  Selected results for the Test dataset, min_support>=0.15, min_confidence>=0.5.

  Number of frequent patterns meeting min_support:

      1 element: 5
      2 elements: 9
      3 elements: 3

  Patterns generated from three-element frequent patterns (i.e., not all rules) exceeding min_confidence:

      2, 5 -> 1 (conf=0.67)
      1, 2 -> 5 (conf=1)
      3, 5 -> 1 (conf=0.67)
      1, 3 -> 5 (conf=0.67)
      4, 5 -> 1 (conf=0.67)
      1, 4 -> 5 (conf=1)
</details>

## Exercise 2
This exercise focuses on practical applications of exploratory data analysis, where students will extract meaningful insights from datasets and evaluate their effectiveness using real-world examples.

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_02.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_02.ipynb)


## Exercise 3
This exercise focuses on implementation of hierarchical clustering techniques, where students will create their own implementation of agglomerative clustering using single and complete linkage methods, and visualize the results using scatter plots.

1. Implement two agglomerative clustering approaches — **single linkage and complete linkage** — using **Manhattan and Euclidean metrics** derived from a distance matrix.
2. **Stop criteria**: either terminate when a predefined number of clusters is reached or carry out full clustering and then cut the dendrogram at the appropriate level to yield the desired number of clusters.
3. **Visualization**: present the resulting cluster assignments using a **scatter plot** to display the clustering structure.

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_03.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_03.ipynb)

## Exercise 4
This exercise focuses on application of clustering techniques on the selected dataset from the Kaggle or UCI repository. Students will apply different clustering algorithms, compare their results, and visualize the clustering structure using scatter plots.

1. Download a dataset from [Kaggle](https://kaggle.com) or [UCI Machine Learning Repository](https://archive.ics.uci.edu).
2. Apply K-means, Hierarchical and DB scan algorithms.
3. Compare the results and interpret the results.

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_04.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_04.ipynb)

For the Exercise 4, there is also an interactive example, where we would demonstrate how can clustering be used in the real-world example, in this case we would try to use clustering on the property data. The example is in the [Examples folder](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/examples/clustering/ml_04)

## Exercise 5
The aim of the exercise is to test the possibilities of dimension reduction and verify the ability of methods to reduce noise in data. Use SVD decomposition and examine the error of matrix reconstruction. Another task is to use PCA and TSNE methods to create a 2D visualization of data.

1. Apply SVD (complete) and Reduced SVD method to Bars Dataset and evaluate also data with noise.
2. Reconstruct the data with different reduction (2, 5, 10, 16 basis functions).
3. Try the NNMF method similarly.
4. Take the Mnist dataset and apply SVD recomposition and visualize it in vector space. Coloring take from Labels
5. Try to reconstruct the reduced data and visualize it.
6. Use PCA and t-SNE for further visualization of the Mnist dataset.

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_05.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_05.ipynb)

For the Exercise 5, there is also an interactive guide with the mathematical explanations to understand the dimension reduction algorithms, intuition behind them and what are they good for. This example is for SVD, NNMF, PCA and t-SNE [Examples folder](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/examples/dimension_reduction/)

## Exercise 6
The aim of the exercise is to test ability to implement decision trees - an elementary algorithm with single splitting criteria that is easy to understand and interpret. Students should be able to apply the concepts learned in the course to build a decision tree from scratch.

1. Implement a decision tree from scratch.
2. Use a simple dataset to train the decision tree.
3. Visualize the decision tree structure (console output is enough).
4. Evaluate the performance of the decision tree.

**Note**: You can also use any dataset from the [ml_06 datasets](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/datasets/ml_06/)

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_06.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_06.ipynb)

## Exercise 7
The aim of the exercise is to test various classification algorithms from Scikit-learn library. Students should be able to apply preprocessing when needed, and compare results of different methods. You may use the prepared template and data.

1. Use classical datasets to train different models.
2. Preprocess the data to gain maximum performance.
3. Apply the classification algorithms from Scikit-learn.
4. Evaluate the performance of the model.

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_07.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_07.ipynb)

## Exercise 8
The aim of the exercise is to test various regression algorithms from Scikit-learn library. Students should be able to apply preprocessing when needed, feature selection, and compare results of different methods. You may use the prepared template and data.

1. Use classical dataset or presented dataset to train different models.
2. Preprocess the data to gain maximum performance.
3. Apply the regression algorithms from Scikit-learn.
4. Evaluate the performance of the model.

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_08.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ml_08.ipynb)


# 💡 Notes
## How to create a Python Virtual Enviroment named `venv`
### Create `venv`
```
python -m venv venv
```
OR
```
python3 -m venv venv
```

### Activate `venv`

* Activate `venv` in **Windows**
```
.\venv\Scripts\Activate.ps1
```

* Activate `venv` in **Linux/macOS**
```
source venv/bin/activate
```

### Create
### Install python packages
- Navigate to the root folder where the **requirements.txt** is located

```
pip install -r requirements.txt 
```

### 🚀 Run Jupyter lab

```
jupyter lab
```

## How to run Jupyter notebook directly in the VSCode
1. Download **Jupyter** extension
2. Follow the [How to create a Python Virtual Enviroment](###Create)
3. Directly open the ml_xx.ipynb in the VSCode
4. Enjoy the local experience

### Hints for the VSCode

- You can use same shortcuts as with the *JupyteLab* or *Google Colab*
- Sometimes the jupyter kernel may freeze, you can use the VSCode command to reload window
  - **MacOS**  ```shift + command + P``` -> Reload Window
  - **Windows/Linux** ```shift + alt + P``` -> Reload Window
- You can use the ```pip install library_name``` directly in the Jupyter notebook

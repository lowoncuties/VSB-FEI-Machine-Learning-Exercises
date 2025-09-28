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

#  Machine Learning
## The course is held at FEI, VSB-TU Ostrava

Course information may be found [here](https://homel.vsb.cz/~pla06/subject_ml.html).

Feel free to contact me (<lukas.jochymek.st@vsb.cz>) if you have any questions or want to discuss any topic from the course 😊

All authorship is mentioned where possible.

## 📚 Study materials

Materials that should help you with the basic concepts from the mathematics and statistics needed by this course.

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Fundamentals-of-Machine-Learning-Exercises/blob/master/statistics_explained.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Fundamentals-of-Machine-Learning-Exercises/blob/master/statistics_explained.ipynb)


Any PR is welcome, whether you find a typo or you have better explanation

# 📊 Exercises
## Exercise 1
The aim of the exercise is to get an overview of the course, get familiar with the jupyter notebooks and be able to setup a Python Virtual Enviroment (`venv`)

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
3. Directly open the fml_xx.ipynb in the VSCode
4. Enjoy the local experience

### Hints for the VSCode

- You can use same shortcuts as with the *JupyteLab* or *Google Colab*
- Sometimes the jupyter kernel may freeze, you can use the VSCode command to reload window
  - **MacOS**  ```shift + command + P``` -> Reload Window
  - **Windows/Linux** ```shift + alt + P``` -> Reload Window
- You can use the ```pip install library_name``` directly in the Jupyter notebook

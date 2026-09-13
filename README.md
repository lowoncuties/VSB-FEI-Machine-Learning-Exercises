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
This exercise focuses on the implementation and understanding of basic neural networks in PyTorch.

> [Jupyter Notebook](https://github.com/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ML_01_Basic_Neural_Network_PyTorch.ipynb)

> [Google Colab](https://colab.research.google.com/github/lowoncuties/VSB-FEI-Machine-Learning-Exercises/blob/main/ML_01_Basic_Neural_Network_PyTorch.ipynb)



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


## 🎥 Additional learning materials

- **StatQuest with Josh Starmer** 
  [ML playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) · [All series, including neural networks and DL](https://statquest.org/video_index.html)

- **3Blue1Brown — Neural Networks** 
  [YouTube playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

- **CS229 — Machine Learning, Andrew Ng (2018)** 
  [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)

- **CS231n — Deep Learning for Computer Vision** 
  [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16) · [Course materials](https://cs231n.stanford.edu/)

- **CS224N — Natural Language Processing with Deep Learning (2024)** 
  [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) · [Course materials](https://web.stanford.edu/class/cs224n/)

- **Andrej Karpathy — Neural Networks: Zero to Hero** 
  [Video series and accompanying code](https://karpathy.ai/zero-to-hero.html)

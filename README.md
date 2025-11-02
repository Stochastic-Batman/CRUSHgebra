# CRUSHgebra
Can AI predict your grades AND your love life?

Multi-task deep learning model for simultaneous prediction of student grades (regression) and relationship status (classification) from the UCI Student Performance dataset. Features a shared MLP body with dual task-specific heads, a custom training loop with combined loss functions, and comprehensive evaluation metrics. Built with PyTorch, it learns from student data to predict both academic performance and romantic status. One brain, two tasks, endless possibilities. Because why choose between math and romance?

## Dataset
For this project, we use the UCI Student Performance dataset. Instructions on how to obtain the data and the methods and return values available are at https://github.com/uci-ml-repo/ucimlrepo.

## Setup
Check your Python version with `python --version`. If it is not already Python 3.14, set it to 3.14. Then create a virtual environment with:

`python -m venv crushgebra_venv`

and install requirements with:

`pip install -r requirements.txt`
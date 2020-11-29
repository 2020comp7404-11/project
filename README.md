# How does feature interaction impact recommendation algorithm
###### COMP7404 Project Group 11

[https://github.com/2020comp7404-11/project](https://github.com/2020comp7404-11/project)

### Setup
- Python 3.7 is needed, if your local python version is different, `virtualenv` should be used:

  ```bash
  # This part of instrcutcion is for OSX
  # Install python 3.7 via homebrew
  brew install python@3.7
  ```
  Init virtualenv in `venv` folder

  ```bash
  pip3 install virtualenv
  virtualenv -p /usr/local/opt/python@3.7/bin/python3 venv
  ```

  Activate the virtualenv
  ```bash
  source venv/bin/activate
  ```
- Install dependencies
  ```bash
  pip install -r requirements.txt #Install dependencies via pip
  ```

### Running the demo
- Run all models and plot the AUC graph
  ```bash
  python demo.py
  ```
- Run individual model to get their AUC output 
  ```bash
  python <model>.py
  ```
  For exmaple, to run xdeepfm
  ```bash
  python xdeepfm.py
  ```

### Dataset
[https://www.kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit)

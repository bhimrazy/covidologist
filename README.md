<p align="center">
  <img src="https://user-images.githubusercontent.com/46085301/183135654-42cb6a4b-bf27-4e10-b2de-e6be196b2b96.png" height="150"/>
  <br/>
A Disease Detection Web App to assist radiologists to detect the presence of COVID-19.
</P>

## Installation
Run my Project
```shell
    # clone the repo and check into the dir
    git clone https://github.com/bhimrazy/covidologist
    cd covidologist
    
    # Setup environment and install all the requirements
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Setup kaggle key or download kaggle.json key file and place it in "~/.kaggle"
    export KAGGLE_USERNAME="your kaggle username"
    export KAGGLE_KEY="your kaggle api key"

    # Download Datasets from kaggle (https://www.kaggle.com/datasets/andyczhao/covidx-cxr2)
    kaggle datasets download -d andyczhao/covidx-cxr2
    
    # unzip to temp folder
    unzip covidx-cxr2.zip -d temp
    
    # remove zip file
    rm -rf covidx-cxr2.zip
    
    
    # prepare dataset folder
    python main.py prepare
    
    # train model
    python main.py train
    
    # generate metrics
    python main.py generate
    
    
    # Run fast api app
    cd app && uvicorn main:app --reload
```

## 📚 RESOURCES:
◆ PyTorch: https://pytorch.org <br/>
◆ FastAPI: https://fastapi.tiangolo.com <br/>
◆ COVIDx CXR-2 Dataset: https://www.kaggle.com/datasets/andyczhao/covidx-cxr2

## Author
- [@bhimrazy](https://www.github.com/bhimrazy)

## Preview
[![covid 19 disease detection](https://user-images.githubusercontent.com/46085301/183138564-bdaaa457-5f31-47e5-889d-f7331a8ffebb.png)](https://covidologist.herokuapp.com/)

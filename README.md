# Data Mining Project
Project of Data Mining course UniTn (Università di Trento) a.y 2020-2021

## Folders Content

```
project
│   README.md
│       
└───doc
│   │   report.pdf
│   
└───src
|   │  project_algorithm.py
|   │  
|   └─ Preprocessing
|
└───data
|   | 
|   └─ csv
|   | 
|   └─ pickle
|
└───bin
    |  results.csv
    |
    └─ Evaluation 
```

## Requirements
```bash
pip3 install -r requirements.txt
```
## Execution
```bash
python3 src/project_algorithm.py 
```
## NOTES!!
* Since the dataset on which the report is based is large (1.6 million tweets) it is possible to decide whether to run the algorithm with this dataset or with the one you provided.
Just select from code-line after launching the program.
The program for the final_dataset takes about 7 minutes, instead of for the dataset you provided 20 seconds maximum.
* Results are present in the bin folder for both the dataset I used and the one you provided.
Also present are the preprocessed datasets of both formats (.csv for visualization and .pkl for execution on code) and again I have kept both the one used for the project and the one you provided.
* All files used for evaluation are inside the bin folder according to what is specified in the report.
* Only the input dataset provided by you is available in the /data/Input. The input dataset used in this project hasn't been upload ONLY because of its huge size (500MB). If you need I can provide that to you immediately.
## Contributor
Elia Cunegatti

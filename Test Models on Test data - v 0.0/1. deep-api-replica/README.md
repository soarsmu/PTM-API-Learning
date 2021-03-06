# Deep API Learning

It is a PyTorch implementation of deepAPI. See [Deep API Learning](https://arxiv.org/abs/1605.08535) for more details. 

## Prerequisites
 - PyTorch 0.4
 - Python 3.6
 - Numpy
 

## Usage

### Run the following command from the current directory for initializaing a virtual environment. This will create a folder, *api-env* in the current directory:

```
python -m venv api-env
source api-env/bin/activate (for linux)
api-env\Scripts\activate (for windows)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Test

Then you can run the model by:

    python sample.py
    
The outputs will be printed to stdout and generated responses will be saved at results.txt in the `./output/` path.




## References 
If you use any source codes or datasets included in this toolkit in your
work, please cite the following paper:
    
    @inproceedings{gu2016deepapi,
        author = {Gu, Xiaodong and Zhang, Hongyu and Zhang, Dongmei and Kim, Sunghun},
        title = {Deep API Learning},
        booktitle = {Proceedings of the 2016 24th ACM SIGSOFT International Symposium on Foundations of Software Engineering},
        series = {FSE 2016},
        year = {2016},
        location = {Seattle, WA, USA},
        pages = {631--642},
        publisher = {ACM},
        address = {New York, NY, USA},
    }
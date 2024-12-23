train: 
	WANDB_API_KEY=$(shell cat .wandbkey) venv/bin/python mandelbrot_setnn.py 

setup_venv:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
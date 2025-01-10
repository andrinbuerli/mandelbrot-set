include resources/common.mk

###########################
# Training
###########################
train: ##@Training train the model, requires .wandbkey file
	WANDB_API_KEY=$(shell cat .wandbkey) venv/bin/python scripts/train_mandelbrot_setnn.py 


###########################
# PROJECT UTILS
###########################

render_mandelbrot: ##@Utils render mandelbrot set (use PARAMS for cli arguments)
	venv/bin/python scripts/render_mandelbrot_set.py $(PARAMS)

setup_venv: ##@Utils setup virtual environment
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
include resources/common.mk

FILE := default.yaml

###########################
# Training
###########################
train: ##@Training train the model, requires .wandbkey file (FILE for cfg file)
	WANDB_API_KEY=$(shell cat .wandbkey) venv/bin/python scripts/train_mandelbrot_setnn.py --config-name $(FILE)


###########################
# PROJECT UTILS
###########################

render_mandelbrot: ##@Utils render mandelbrot set (use PARAMS for cli arguments)
	venv/bin/python scripts/render_mandelbrot_set.py $(PARAMS)

render_checkpoint: ##@Utils render mandelbrot set using neural network (use PARAMS for cli arguments)
	venv/bin/python scripts/render_checkpoint.py $(PARAMS)

setup_venv: ##@Utils setup virtual environment
	python3 -m venv venv
	venv/bin/python -m pip install -e .
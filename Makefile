# define the name of the virtual environment directory
VENV := venv

# default target, when make executed without arguments
all: venv

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install wheel
	./$(VENV)/bin/pip install -r requirements.txt
	./$(VENV)/bin/pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html 

# venv is a shortcut target
venv: $(VENV)/bin/activate

run: venv
	$(VENV)/bin/python3 ./run.py $(VENV)/bin /bids_dir /out_dir participant --gpuid ${GPUID}
	$(VENV)/bin/python3 ./run.py $(VENV)/bin /bids_dir /out_dir group --gpuid ${GPUID}

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete

.PHONY: all venv run clean


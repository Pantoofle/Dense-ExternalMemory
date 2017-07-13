IGNORE=$(shell cat .gitignore | grep -v '\#' | tr '\n' ' ')
DEPS=$(shell cat dependancies.txt)

LALF_EXE=online
LALF_DIR=libalf/

CPP=g++
C_FLAGS=-lalf
LD_LIBS=LD_LIBRARY_PATH=/usr/local/lib 

all: run

online: $(LALF_DIR)online.cpp
	$(CPP) $(LALF_DIR)$(LALF_EXE).cpp $(C_FLAGS) -o $(LALF_DIR)$(LALF_EXE)

pip_deps:
	pip install --upgrade $(DEPS)

build: online
	mkdir -p models
	mkdir -p img

run: build
	rm -rf logs/
	TF_CPP_MIN_LOG_LEVEL=1 python main.py

clean:
	@echo Cleaning...
	rm -rf $(IGNORE)

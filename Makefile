IGNORE=$(shell cat .gitignore | grep -v '\#' | tr '\n' ' ')

build:
	mkdir models

run:
	rm -rf logs/
	TF_CPP_MIN_LOG_LEVEL=1 python main.py

clean:
	@echo Cleaning...
	rm -rf $(IGNORE)

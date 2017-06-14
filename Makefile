IGNORE=$(shell cat .gitignore | grep -v '\#' | tr '\n' ' ')

build:
	mkdir models

run:
	rm -rf logs/
	python main.py

clean:
	@echo Cleaning...
	rm -rf $(IGNORE)

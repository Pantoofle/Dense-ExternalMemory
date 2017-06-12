IGNORE=$(shell cat .gitignore | grep -v '\#' | tr '\n' ' ')

run:
	python main.py

clean:
	@echo Cleaning...
	rm -rf $(IGNORE)

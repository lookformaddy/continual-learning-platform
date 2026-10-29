.PHONY: test demo eval lint

test:
	PYTHONPATH=src pytest -q

demo:
	PYTHONPATH=src python -m continual demo

eval:
	PYTHONPATH=src python -m continual eval

lint:
	python -m compileall -q src tests





all:
	python setup.py build_ext --inplace


clean:
	rm -rf tree/*.so tree/*.pyc tree/*.c


test:
	py.test test_*

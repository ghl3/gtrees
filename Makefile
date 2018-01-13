

all:
	cython

cython: tree/*.pyx tree/*.pxd
	python setup.py build_ext --inplace

clean:
	rm -rf tree/*.so tree/*.pyc tree/*.c *.pyc

test: cython
	py.test test_*

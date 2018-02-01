all: unzip run

unzip: train-images-idx3-ubyte train-labels-idx1-ubyte

train-images-idx3-ubyte: train-images-idx3-ubyte.gz
	@cp $< tmp.gz
	@gunzip tmp.gz
	@mv tmp $@

train-labels-idx1-ubyte: train-labels-idx1-ubyte.gz
	@cp $< tmp.gz
	@gunzip tmp.gz
	@mv tmp $@

train-images-idx3-ubyte.gz:
	@wget http://yann.lecun.com/exdb/mnist/$@

train-labels-idx1-ubyte.gz:
	@wget http://yann.lecun.com/exdb/mnist/$@

run:
	@./learning.py

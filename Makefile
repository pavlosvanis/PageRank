basic:
	rm -f basic.npz basic.npy
	python3 pagerank.py basic.adjlist 10

stanford:
	python3 pagerank.py web-Stanford.adjlist 10

berkstan:
	python3 pagerank.py web-BerkStan.adjlist 10

google:
	python3 pagerank.py web-Google.adjlist 10

destroy:
	rm -f *.npz *.npy
	rm -rf __pycache__

fullrun: stanford berkstan google

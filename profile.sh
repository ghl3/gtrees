
python -m cProfile -s time -o script.profile test_fit.py --max-depth 4 --num-samples 1000 --num-iterations 3 | tee profile.txt
pyprof2calltree -i script.profile -o script.calltree
qcachegrind script.calltree

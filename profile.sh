
python -m cProfile -s time -o script.profile test_fit.py --leaf-prediction logit --max-depth 4 --num-samples 10000 --num-iterations 1 | tee profile.txt
pyprof2calltree -i script.profile -o script.calltree
qcachegrind script.calltree

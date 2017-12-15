
python -m cProfile -s time -o script.profile profile.py --loss cross_entropy --leaf-prediction logit --max-depth 3 --num-samples 5000 --num-iterations 1 2>&1 | tee profile.txt
pyprof2calltree -i script.profile -o script.calltree
qcachegrind script.calltree

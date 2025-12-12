# This is a runner script to execute all algorithms on both the nightbeatdb and aw dataset.
# Increase the number of workers if your system allows
python align.py --datasets nightbeatdb aw --workers 4
python transform.py --algorithm all --datasets nightbeatdb aw --workers 4
python predict.py --algorithm all --datasets nightbeatdb aw --workers 4
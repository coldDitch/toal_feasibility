# Examples of toal functions

## Getting started

- Create new python environment or use existing one
- Activate environment and install requirements by
``` pip install -r requirements.txt ```

or alternatively using virtualenv

- Create folders for temporary files and install python environment
(Helper)
``` source setup.sh ```

## Running tests
- Linear example
	- The active learning for the functions can be run by:

	```python activelearning.py [model_function] [train_n] [test_n] [path] [seed] [acquisition] [queries] ```

	eg. 
	
	```python activelearning.py multilin 6 6 ./ 1234 decision_ig 1```

	```python activelearning.py multilin 20 20 ./ 4321 random 5```

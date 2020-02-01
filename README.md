# Censored regression with parametric models

## Getting started

- Create new python environment or use existing one
- Activate environment and install requirements by
``` pip install -r requirements.txt ```

- Create folders for temporary files such as stan binaries and plots
(Helper)
``` sh testruns/setup.sh ```

## Running tests
- Linear example
	- The active learning for the functions can be run by:

	```python activelearning.py [model_function] [train_n] [test_n] [path] [seed] [acquisition] [queries] ```

	eg. 
	
	```python activelearning.py test_case 6 6 ./ 1234 avgtoal 1```

	```python activelearning.py rbf 20 20 ./ 1234 eig 5```

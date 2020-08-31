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
- Set up config file for the test. For more information check the comments in the config file.

- Running code 
	- The active learning for the functions can be run by:

	```python activelearning.py [path] [seed] [acquisition] [queries] ```

	eg. 

	```python activelearning.py ./ 4321 random 5```
	
	```python activelearning.py ./ 1234 decision_ig 1```

- Running on triton
	- Modify array length to obtain more runs

	- Change acquisition and query numbers
	```sbatch run_triton.sh```
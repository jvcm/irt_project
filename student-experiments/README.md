## SETUP ##
To run the code needs these python packages:
	 tensorflow,
	 edward,
	 numpy,
	 pandas,
	 matplotlib,
	 seaborn.


## Test steps ##

1. Generate IRT response data by ```gen_irt_data.py```, for example:
        
		python gen_irt_data.py dataset:moons noise_fraction:0.2 random_seed:1
        
	
	* The full usage of this script will display if run it without args and an example data file will be generated:

		    python gen_irt_data.py 

	* There are 3 synthetic datasets supported in the script: moons, clusters, circles, and 2 real datasets: mnist, fashion
	* The classifiers need to be selected in the script (classifiers dictionary in the begining)
	* The script will generate 2 csv files: 
		1. ```irt_data_*.csv```: the response (predicted probability) of all test data from all classifiers. 
		2. ```xtest_*.csv```: the features (the coordinates of x and y axis) of test data, and a bool feature 'noise' to indicate if a data point has a wrong label.
	
2. Train the Beta_IRT model by ```betairt_test.py```, for example, the irt_data file from previous step is ```'irt_data_moons_s400_f20_sd1_m12.csv'```:

		python betairt_test.py irt_data_moons_s400_f20_sd1_m12.csv a_prior_std:1.5
		
	* The full usage of this script will display if run it without args:

			python betairt_test.py

	* The script will generate 3 figures and 3 csv files:
		1. ```irt_parameters_*.csv```: the difficulty and discrimination of test data points
		2. ```irt_ability_*.csv```: the ability of classifiers, the last row is the standard deviation of all abilities
		3. ```dnoise_performance_*.csv```: the precision and recall of the Beta_IRT model for locating label noise
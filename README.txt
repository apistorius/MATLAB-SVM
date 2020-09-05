HOW TO RUN MY SVM MODEL SCRIPT:
-------------------------------

1. Make sure the data is within the same folder as assignment.m

2. Change the filename on line 8 to the filename of the desired data.

3. Select the number of K-Folds on line 22

4. Indicate the starting location for the testing data with in the K-Fold
	-  This is the first observation in the data set that will become a testing point.
	   the next (k/n) will also be test data
        

        ====================================================================================
	|       training data           |  testing data   |        training data           |
	====================================================================================
                                        ^
                                        |
				    test_begin


5. Change the functions between the dual/primal versions to swap between forms.

6. Run and see the results.
PART 4:
 
We have separated our code into 2 files, namely part4fun.py and part4.py. 

These files are:

- part4fun.py:  This .py file contains the functions written for and utilized in part4.py.  
		This file isimported in part4.py.
- part4.py:     This.py contains the ‘main’ code block used to run the transition and emission parameter
		generation, and second-order Viterbi algorithm to predict the tags for the test data.
		
To change the test data, you can simply edit the code block entitled ‘validation set’ (2) and run the code,
which will also print the results/scores. part4.py can also be run simply on the python notebook entitled
test.ipynb.


PART 5 (PERCEPTRON):

script named perceptron can be found in folder part5
python perceptron.py -traindata ../FR/train -testdata ../FR/dev.in -output ../FR/test.p5.out -k 1 -epochs 54

- traindata, location of language file to train on eg ../EN/train
- testdata, location of language test file to train on eg ../EN/dev.in
- output, location of language test file to train on eg ../EN/test.p5.out
- k, parameter to remove words that occur less the k times
- epochs, number of epochs to train the perceptron 



PART 5 (HMM):

This is for the part 5 portions that are only modifications to the HMM model. 
Similar to part 4, we have separated our code into 2 files, namely part5fun.py and part5.py. 

These files are:

- part5fun.py:  This .py file contains the functions written for and utilized in part5.py.  
		This file isimported in part5.py.
- part5.py:     This.py contains the ‘main’ code block used to run the transition and emission parameter
		generation, and second-order Viterbi algorithm to predict the tags for the test data.
		
To change the test data, you can simply edit the code block entitled ‘validation set’ (2) and run the code,
which will also print the results/scores. part5.py can also be run simply on the python notebook entitled
test.ipynb.


Capstone project directory

Contains python code and ipython notebooks

Contains the PDF for proposal and final report


Instructions for running the scripts
=====================================

1) Create a top level directory
2) Create a 'data' directory under it and copy training files and test files under it
   The following should be the dir structure - 


	TOP LEVEL DIR
		DIR 'data'
		|
		----- DIR 'train-jpg' = keep all images for training
		|
		------DIR 'test-jpg'  = keep all images for testing
		|
		------DIR 'test-jpg-additional' = keep additonal images for testing


		DIR notebooks 
		|
		------------- AmazonPredict.ipynb  = This is for creating result  and final submission file
		|
		--------------pre_process.ipynb = run this first to pre-process images
		|
		--------------NewKerasTrain.py  = for training the model
		|
		--------------NewKerasTest.py = for predicting results against test images
                |
                --------------NewKerasValidate.py  = for fine tuning threshold and calculating F1 score


3) Following is the order for running the scripts 
		* Run pre_process.ipynb to pre-process training and test images
		* Run NewKerasTrain.py for training the model
		* Run NewKerasValidate.py to calculate the threshold for labels
		* Run NewKerasTest.py to generate output labels against test images
		* Run AmazonPredict to generate output labels and visualizations. 
			(This will also create Kaggle final submission file)

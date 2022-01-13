Read-Me for the Code
A short information on how to use run the Modell is given here. 

BERT Modell

Preprocessing

The Processing Pipeline starts with the File 

‘Training_data’, which is processed in the 
‘Data_Preparation.py’ class. The output of this class is called 
‘Prepared_data’ and is processed in 
‘Preprocessing_Pipeline.py’. The output here is called 
‘Pre_processed_data’. In this file, I showcase some preprocessing methods and I also save the stopwords in a file called 
‘nltk_german_stop_words.txt', however, this one is not processed any further. 
‘Bert_preprocess.py’ includes some experimental pre-processing before starting model implementation and can even be ignored in this pipeline. 


Modell Implementation

The BERT Modell is implemented in 
‘BERT_training.py’ with the 
‘Prepared_data’ file, 

where it finally does output the classification results. 

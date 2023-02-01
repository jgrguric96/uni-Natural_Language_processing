#NLP Assignment 1:
###Authors: 
- Balázs Borsos
- Dominique Budding 
- Josip Grgurić

###Code parts:
 As requested, the code is separated into three parts: partA, partB and partC.
 There is the SemEval2018-Task3 folder which is used to gain access to the datasets necessary in order to train and test the data
 And finally there is the model_save folder which contains a saved model used in subtask C 
###How to run the code:
 In order to install the requirements necessary to run the code you must first run the following command:
 ```pip install -r requirements.txt```
 
 Once the requirements are installed you should be able to run any of the parts of the NLP assignment.
 
 ###Info:
 For partA and partB there shouldn't be any problems in running the code, however partC is a bit different in that regard.
 
 With partC, we have noticed one of two things can happen:
 - Code will attempt to run torch using a user's CPU instead of GPU which will not be able to handle the workload necessary 
 - Code will not be able to load the saved model

 In the first case, the reason this happens is due to an issue with installing pyTorch with cuda. 
 In case of such a problem the following command can be run in order to manually install pyTorch with cuda:
 ```pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html```
 
 The second problem is because the current process ( subtasks a & b ) must first finish prior to attempting access to the model_save folder. 
 This problem was resolved by doing the final subtask first ( subtask c ). 
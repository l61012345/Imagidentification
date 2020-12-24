# citp
the project is named as citp using Paddlepaddle platform to train pics and identify them using MobileNetV1.   
* **createdatelist**  
 the a list of the pics dataset will be generated  
 * **mobilenetv1**  
 the file to define the algorithms  
 * **train**  
 the program to build the model 
 >you can use train-timestamp to pin each time for each 100 batch in default 
 * **process**  
 preprocess using gauss filters or converting to hsv using process-hsv
 * **infer**  
 the file to identify the pic  
 > you can use infer-multi to indentify new pics in one folder   
* **gui**  
it can realize the identifacation using screenshot with a gui  
as it says that, it is really primitive and written for just "we seem to need a gui"

# Imag identification
the project uses Baidu Paddlepaddle platform to train pics and identify them using MobileNetV1 and other algorithms.   

## file list
* **createdatelist**  
 the a list of the pics dataset will be generated. In addtion, the pics are supposed to be stored in different folders divided by types. 
 * **mobilenetv1**  
 the file to define the algorithms  
 * **train**  
 the program to build the model 
 >you can use train-timestamp to pin each time for each 100 batch in default 
 * **process**  
 preprocess using gauss filters or converting to hsv using process-hsv
 * **infer**  
 the file to identify the pic  
 > you can use infer-multi to indentify a batch of new pics in the same folder   
* **gui**  
it can realize the identifacation using screenshot with a gui  
as it says that, it is really primitive and written for just "it seems that we need a gui"


# Imag identification
the project uses Baidu Paddlepaddle platform to train pics and identify them using MobileNetV1 and other algorithms (see other branches).  

## file list
* **createdatelist**  
 generate the list of dataset. In addtion, the pics should be stored in different folders according to their types before running it. 
 * **mobilenetv1**  
 define the algorithm, which is mobilenet v1.
 * **reader**
 preprocess the images from dataset
 * **train**  
 the program to build the model using CPU or GPU
 >you can use train-timestamp to pin each time for each 100 batch in default 
 * **process**  
 preprocess using gauss filters or converting to hsv space using process-hsv
 * **infer**  
 the file to identify the pic  
 > you can use infer-multi to indentify a batch of new pics in the same folder   
* **gui**  
it can realize the identifacation using screenshot with a gui  
as it says that, it is really primitive and written for just "it seems that we need a gui"


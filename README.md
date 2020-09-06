# document-scanner



This project is made with Python 3.x and OpenCV, Numpy , Matplotlib libraries. 


Using this code, it's posible to scan documents by taking a photo in any of standard formats such as .JPG, .PNG, .JPEG, etc. and remove coffee or pen spots. 

**USAGE**

```
pip install matplotlib
pip install numpy
pip install opencv-python
```

If you want to get the intermediate results with the final result :
```
python3 document-scanner.py -S <your-file>
```
If you want to  get original image and final result :
``` 
python3 document-scanner.py -F <your-file>
```

If you want only the final result :
```
python3 document-scanner.py -R <your-file>
```



# Winter Semester 2021 - Project

## to Run the detectors:
1. Prepare all files:
    1. Unzip the _Contour_Detectors.zip_
    2. Make sure that both folders exists in the main directory
    3. Download The encoder-decoder network weights from this link [Models Weights](https://drive.google.com/file/d/1vRqmVjxqkYLMHvPjw8-uny2gxGWWpZuc/view?usp=sharing "Models Weights")
    4. In the _ContourDetector.py_ in the _Contour-Detection-Pytorch_ edit the path (in line 26) of the model to the path of the modelweights downloaded. 
2. Now the detectors are ready, open text.py File in the main directory
4. in the Main, change the path to be the path of the image to be tested.
    * make sure the image path is in the format `r"D:\path-to-the-image"`
5. in the command line run the bellow line OR run the text.py file using any editor
    `python test.py`

- - - -
## results example:

### The Image
![picture alt](https://github.com/BerlinDMET901/project-mit-kase-1/blob/main/8068.jpg "The Image")

### The results:

![picture alt](https://github.com/BerlinDMET901/project-mit-kase-1/blob/main/Results.png "The results")

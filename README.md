# The MINERVA Dataset: 
**Benchmarking the detection of musical instruments in unrestricted, non-photorealistic images from the artistic domain**

This repository contains ...

![alt text](https://github.com/paintception/MINeRVA/blob/master/images/readme_img.png)


* Training `./train`

* If you would like to test an already trained YOLO-V3 model the first step is to 
download our pre-trained weights from [here](). Once this is done you will find all 
the necessary testing code in `./test/`. We provide a simple example that should allow you
to detect a Lute in the testing image we provide in `./testing_images`. As visually
represented in the figure below.

    * Once you have downloaded the pre-trained models be sure to pass the correct `.h5` 
    file to the `yolo.py` script. The same script will also require a path to a file 
    containing the anchors that are used to detect bounding boxes, and also to another 
    file that contains which classes the model is supposed to detect. For this example
    you can find the anchors file in `./anchors/` and the file containing the classes that
    need to be detected in `./classes/`. Once this is done the only thing left to do is to
    simply run `python test_yolo.py --test`.
    
    ![alt-text-1](https://github.com/paintception/MINeRVA/blob/master/testing_images/Q46580615.jpg)  ![alt-text-1](https://github.com/paintception/MINeRVA/blob/master/test/detected_images/detections_Q46580615.jpg)
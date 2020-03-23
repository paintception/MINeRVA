# MINERVA
## Benchmarking the detection of musical instruments in unrestricted, non-photorealistic images from the artistic domain

### Scope
This repository contains all the code that can be used to replicate the results presented in the [journal paper]() 
> Matthia Sabatelli, Nicolae Banari, Marie Cocriamont, Eva Coudyzer, Karine Lasaracina, Walter Daelemans, Pierre Geurts & Mike Kestemont, "Advances in Digital Music Iconography. Benchmarking the detection of musical instruments in unrestricted, non-photorealistic images from the artistic domain". *Digital Humanities Quarterly* (2020).

![alt text](https://github.com/paintception/MINeRVA/blob/master/images/readme_img.png)

We would like to start by acknowledging GitHub users with their repositories that have all 
served as more than a solid starting codebase for what is presented hereafter. Specifically most of the code
that is related to the object-detection experiments comes from [this]() GitHub repo!

### Codebase:

* Data formatting: `./prepare_data/`


* Training: `./train/`

* Testing: if you would like to **test** an already trained YOLO-V3 model the first step is to 
download our pre-trained weights from [here](). Once this is done you will find all 
the necessary testing code in `./test/`. We provide a simple example that should allow you
to detect some of the instruments that are part of the testing set of the top-5 benchmark 
introduced in the paper.

    * Once you have downloaded the pre-trained models be sure to pass the correct `.h5` 
    file to the `yolo.py` script. The same script will also require a path to a file 
    containing the anchors that are used to detect bounding boxes, and also to another 
    file that contains which classes the model is supposed to detect. For this example
    you can already find the anchors file in `../anchors/` and the file containing the classes that
    need to be detected in `../classes_to_detect/`. Once this is done the only thing left to do is to
    simply run `python test_yolo.py --test` and you should obtain the images that are represented 
    in the second row of the following figure.
    
![alt text](https://github.com/paintception/MINeRVA/blob/master/images/detections_examples.jpg)

### License:
The code is distributed under a Creative Commons (CC-BY) licence. Please cite the original paper when using, repurposing or expanding data and/or code.
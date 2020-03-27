# MINERVA
## Benchmarking the detection of musical instruments in unrestricted, non-photorealistic images from the artistic domain

### Scope
This repository contains all the code that can be used to replicate the results presented in the [journal paper]() 
> Matthia Sabatelli, Nikolay Banar, Marie Cocriamont, Eva Coudyzer, Karine Lasaracina, Walter Daelemans, Pierre Geurts & Mike Kestemont, "Advances in Digital Music Iconography. Benchmarking the detection of musical instruments in unrestricted, non-photorealistic images from the artistic domain". *Digital Humanities Quarterly* (2020).

![alt text](https://github.com/paintception/MINeRVA/blob/master/images/readme_img.png)

We would like to start by acknowledging GitHub users [allanzelener](https://github.com/allanzelener),
and [qqwweee](https://github.com/qqwweee) for their wonderful open-source implementations of 
YOLO-V3 object detectors. We directly refer to these two repositories which have both served as more 
than a solid start for the work presented in our paper.

* [YAD2K](https://github.com/allanzelener/YAD2K)
* [keras-yolo3](https://github.com/qqwweee/keras-yolo3)


### Detection:

* Data formatting: the official dataset splits which you can download from [here]() come in the form of the following table
    ```  
       |  image_filename  |  id         |  bounding_box_coordinates  |  instrument_name  |  area_of_bounding_box
    ---|  ----------------|-------------|----------------------------|-------------------|----------------------
    0  |  2921.jpg        |  125386343  |  372,357,663,705           |  Harp (3285)      |  100933.0
    1  |  2921.jpg        |  125386343  |  401,398,631,652           |  Harp (3285)      |  58338.0
    2  |  3405.png        |  125386343  |  436,451,585,580           |  Violin (3573)    |  19279.0
    3  |  3405.png        |  125386343  |  467,480,566,576           |  Violin (3573)    |  9524.0
    4  |  3388.jpg        |  125386343  |  39,365,107,480            |  Lute (3394)      |  7753.0
    ```

    In order to train the object detector we need to extract the bounding-box coordinates and convert 
    each instrument_name to a categorical value. You can find the code to do this in `./prepare_data/`. 
    Simply run the `format_minerva.py` script which will convert the original splits into a file
    which keeps for each instance in the dataset the bounding box coordinates and converts the respective instrument name
    to a categorical value.
    
    ```
    /path/to/your/file/2921.jpg 372,357,663,705,1
    /path/to/your/file/2921.jpg 401,398,631,652,1
    /path/to/your/file/3405.png 436,451,585,580,4
    /path/to/your/file/3405.png 467,480,566,576,4
    /path/to/your/file/3388.jpg 39,365,107,480,2
    ```
    The script will also create a `list_of_instruments.txt` file which will contain the names of the instruments that
    are associated to the different categorical values.

* Training: if the dataset has been properly formatted and therefore comes in the form `/path/to/your/file/2921.jpg 372,357,663,705,1`
   you are ready to **train** your own object-detector. You can find the code to do so in `./train/` and use
   the `train_yolo_model.sh` script. The script will get as input the files that correspond to the different
   annotations, a file containing the instruments that will have to be detected and an anchors file which
   we already provide in `./anchors/minerva_anchors.txt`. The trained models will be stored in `./train/weights`.
    
* Testing: if you would like to **test** an already trained YOLO-V3 model the first step is to 
download our pre-trained weights from [here](). Once this is done you will find all 
the necessary testing code in `./test/`. We provide a simple example that should allow you
to detect some of the instruments that are part of the testing set of the top-5 benchmark 
introduced in the paper.

    * Once you have downloaded the pre-trained models be sure to pass the correct `.h5` 
    file to the `yolo.py` script. The same script will also require a path to a file 
    containing the anchors that are used to detect bounding boxes, and also to another 
    file that contains which classes the model is supposed to detect. For this example
    you can again find the anchors file in `../anchors/` and the file containing the classes that
    need to be detected in `../classes_to_detect/`. Once this is done the only thing left to do is to
    simply run `python test_yolo.py --test`. This will create two new folders: `detected_files`
    and `detected_images`. In the first one you will find a different `.txt` file per image which will report
    the predictions of the model, while in the second folder you will find the images that are represented 
    in the second row of the following figure.

![alt text](https://github.com/paintception/MINeRVA/blob/master/images/detections_examples.jpg)

### Classification:
* The code related to classification part is stored in `./classification_of_crops/`
* Data formatting: in order to train the classifier we need to extract the bounding-box from the image and format it to 
    the specific size. Run the following script to do this:  `python ../cut_images.py -data /path/to/raw/images -splits /path/to/splits -save /path/to/save`. 
    This path `/path/to/splits` must contain 3 files:`train.txt`, `dev.txt` and `test.txt`.  In addition, the script will make one separate path for each class in the corresponding train/dev/test path. 
    You will find the images similar to ones presented in the following figure.
    
![alt text](https://github.com/paintception/MINeRVA/blob/master/images/classification.png)
    
* Training:  the pre-trained models will be stored in `../ECCVModels/`. 
The following pre-trained models are available for fine-tuning: V3, VGG19, ResNet. Run the following script to fine-tune the model of your choice :
`python ../train_an d_predict.py -data /path/to/formatted/data -model_path /path/to/pretrained/model -net name_of_network -save /path/to/save -lr learning_rate`
* Testing: if you would like to **test** an already fine-tuned classifier you should run the following script
`python ../train_and_predict.py -data /path/to/formatted/data -model /path/to/fine-tuned/model `

* Saliency maps: if you would like to get saliency maps for VGG19 you should run the following script 
 `python ../saliency_maps_vis.py -image /path/to/formatted/images -model /path/to/fine-tuned/model -save /path/to/save`

![alt text](https://github.com/paintception/MINeRVA/blob/master/images/sal_map.jpg)



### License:
This work is licensed under a [Creative Commons Attribution 4.0 International
License][cc-by]. Please cite the original paper when using, repurposing or expanding data and/or code.

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
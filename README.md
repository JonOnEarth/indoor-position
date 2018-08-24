# Indoor-position

Information Processing Lab from Northeastern University

Author: Peng Wu, Ni Ke 


Report: https://github.com/JonOnEarth/indoor-position/blob/master/indoor_position_1.pdf

## Introduction
Indoor positioning has been improved a lot and has recently witnessed an increasingmarket interest due to wide-scale use of Internet of Things (IOT) such as smartphonesand other wireless devices in the last couple of years

There are many Indoor positioning techniques, we are using fingerprint positioning technique based on WiFi. It is one of the mostpopular schemes. It contains offline stage and online stage. In the offline stage, thesystem builds a database of thorough measurements from reference locations in targetarea. Then, in the online stage, the system will take the data to its model based onits database to predict the real-time location. Most of existing indoor fingerprintingsystems exploit WiFi RSS values as fingerprints because of its simplicity and lowhardware requirements

## Our work
1. Separate floor detection and position regression as two part;
2. Implement WKNN in matlab&DNN in jupyter notebook;
3. implement ensemble bagging method to get a 100%floor detection;
4. implement stack method to get better result.
5. Compare results of long-term data with the short-term data. Provide a CNN based approach for long-term data

## Data
[Dataset1](https://zenodo.org/record/889798#.WvsnbogvzD4)  
[Dataset2](https://zenodo.org/record/1066041)  
The two datasets are from the same research team, Tampere University of Technol-ogy and Universitat Jaume I. All the data are Wi-Fi database collected in a fullcrowdsourced mode (i.e., different devices, different users and no main indications)

## Environment configuration
tools and frameworks: Python Notebook, Keras, Tensorflow, Matlab

## Code
code 1_2, 1_3 is for floor classification, 2_1, 2_2, 2_3 for floor regression.

long term file is for dataset 2

Matlab file is the code of WKNN using for the dataset.

For more detail, you can check the report.


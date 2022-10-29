# YOLO-V3 Implimentation

In this repository, an implementation of YOLO-V3 is provided. The implementation is based on the paper [YOLO-V3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) by Joseph Redmon, Ali Farhadi.
<!-- The implementation is done using both [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/). -->

## Training Datasets
The training datasets used are the [COCO 2017](https://cocodataset.org/#home) dataset and the [VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset. I've used the kaggle datasets for both of these datasets provided by Aladdin Persson. The links to the datasets are as follows:

 * [MS-COCO](https://www.kaggle.com/datasets/79abcc2659dc745fddfba1864438afb2fac3fabaa5f37daa8a51e36466db101e)
 * [Pascal VOC](https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video)

## Progress Table
<!-- * ✅ - Completed
* 🟡 - In Progress
* ❌ - Incomplete -->

| Syntax      | Progress |
| ----------- | ----------- |
| [TensorFlow Implimentation ](yolov3_tf/Readme.md)   | 🟡 |
| Pytorch Implimentation       | ❌ |
<!-- | Training on Pascal Voc Data  | ❌ | -->
<!-- | Training on MS-COCO Data     | ❌ | -->

## YOLOv3 paper
The code implementation is based on the [YOLO-V3](https://arxiv.org/abs/1804.02767) paper by Joseph Redmon and Ali Farhadi.

#### Please Note:
* In the original YOLO-V3 paper, things were somewhat unclear to me since they were not well defined as the previous(YOLO Family) papers. In some places, I referred to previous papers.

* I've also taken inspiration from [Aladdin Persson](https://www.linkedin.com/in/aladdin-persson-a95384153/)'s [YT Video](https://www.youtube.com/watch?v=Grir6TZbc1M) and [Ayoosh Kathuria](https://www.linkedin.com/in/ayoosh-kathuria-44a319132/)'s [blog(s)](https://www.kdnuggets.com/2018/05/implement-yolo-v3-object-detector-pytorch-part-1.html). I also made some assumptions and changes.

<!-- I'll be explaining each and everything -->


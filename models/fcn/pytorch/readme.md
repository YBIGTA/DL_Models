# PyTorch Implementation (WIP)

* Usage
~~~
python trainer.py \
-- data HYBRID \
-- data_path \path\to\sbd\
-- val_data_path \path\to\VOCdevkit\
-- model_name fcn8s \
-- mode finetuning \
-- optimzizer Adam \
-- lr 1e-4 \
-- n_epoch 200 \
-- check_step 20 \
-- batch_size 32 \
~~~

* Dataset: [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) & [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
  * train: SBD "train.txt" (8498)
  * test:  "val_no_sbd.txt"(904)
    * Images of VOC2012 "val.txt" that is not included in SBD "train.txt"
  
* Result
  * Pixel Accuracy
    * train: 0.9494
    * test: 0.8951
  * train-loss: 0.187611948

* Samples

| input image | output mask |
| ----------- | ----------- |
| ![01_input](results/01_input.png) | ![01_output](results/01_output.png) |
| ![02_input](results/01_input.png) | ![02_output](results/01_output.png) |
| ![03_input](results/01_input.png) | ![03_output](results/01_output.png) |
| ![04_input](results/01_input.png) | ![04_output](results/01_output.png) |
| ![05_input](results/01_input.png) | ![05_output](results/01_output.png) |
| ![06_input](results/01_input.png) | ![06_output](results/01_output.png) |
| ![07_input](results/01_input.png) | ![07_output](results/01_output.png) |
| ![08_input](results/01_input.png) | ![08_output](results/01_output.png) |

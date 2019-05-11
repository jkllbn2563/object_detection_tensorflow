# object_detection_tensorflow

## label tool download
```
指令 

cd ~/code/python/tensorflow 

mkdir addons 

cd addons 

git clone https://github.com/tzutalin/labelImg.git 

cd labelImg 

sudo apt-get install pyqt5-dev-tools 

sudo pip3 install -r requirements/requirements-linux-python3.txt 

make qt5py3 
```
## label
```
指令 

cd ~/code/python/tensorflow/addons/labelImg 

python3 labelImg.py ../../workspace/training_demo/images/train
```
## turn the xml to csv
```
python3 xml_to_csv.py -i $HOME/code/python/tensorflow/workspace/training_demo/images/train -o $HOME/code/python/tensorflow/workspace/training_demo/annotations/train_labels.csv
```
## build label_map
```
item { 

    id: 1 

    name: 'fall' 

} 
item { 

    id: 2

    name: 'box' 

} 
```
## change some detail
```
vim generate_tfrecord.py 
# TO-DO replace this with label map 

# for multiple labels add more else if statements 

def class_text_to_int(row_label): 

    if row_label == 'fall':  # 'ship': 

        return 1 
    elif row_label =='box':
    	return 2
    else:
    	return 0
```

## turn csv to tfrecord

```
python3 generate_tfrecord.py --csv_input=$HOME/code/python/tensorflow/workspace/training_demo/annotations/train_labels.csv --output_path=$HOME/code/python/tensorflow/workspace/training_demo/annotations/train.record --img_path=$HOME/code/python/tensorflow/workspace/training_demo/images/train
```
##download the pre-trained-model
```
指令 

cd ~/code/python/tensorflow/workspace/training_demo/pre-trained-model 

wget -c http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz 

tar zxvf ssd_inception_v2_coco_2018_01_28.tar.gz 

mv ssd_inception_v2_coco_2018_01_28/* . 

rm -r ssd_inception_v2_coco_2018_01_28 
```
##training configuration
```
指令 

cd ~/code/python/tensorflow/workspace/training_demo/training 

cp ~/code/python/tensorflow/models/research/object_detection/samples/configs/ssd_inception_v2_coco.config . 

vim ssd_inception_v2_coco.config（部分編修） 

part1 

model { 

    ssd { 

        num_classes: 1 # Set this to the number of different label classes 

        box_coder { 

            faster_rcnn_box_coder { 

                y_ 

part2 

      decay: 0.9  

      epsilon: 1.0  

    }    

  } 

  fine_tune_checkpoint: "pre-trained-model/model.ckpt" 

  from_detection_checkpoint: true 

part3 

    # effectively bypasses the learning rate schedule (the learning rate will 

    # never decay). Remove the below line to train indefinitely. 

    num_steps: 50 

    data_augmentation_options { 

        random_horizontal_flip { 

        } 

    } 

 part4 

train_input_reader: { 

    tf_record_input_reader { 

        input_path: "annotations/train.record" # Path to training TFRecord file 

    } 

    label_map_path: "annotations/label_map.pbtxt" # Path to label map file 

} 

part5 

eval_input_reader: { 

    tf_record_input_reader { 

        input_path: "annotations/train.record" # Path to testing TFRecord 

    } 

    label_map_path: "annotations/label_map.pbtxt" # Path to label map file 

    shuffle: false 
```
## strat to train
```
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config 

if you already have the different finetune record, please remove it
```
### visualize
```
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-50 --output_directory trained-inference-graphs/output_inference_graph_v1.pb 
```
### test
```
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'img_{}.jpg'.format(i)) for i in range(1, 3) ] 

so you should rename the image to the standard img_?.jpg form and change the number

python3 test_folder.py

cd ~/code/python/tensorflow/workspace/training_demo/detection_result 

eog 0.png && eog 1.png 
```
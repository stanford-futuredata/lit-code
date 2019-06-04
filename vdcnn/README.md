# LIT: Learned Intermediate Representation Training

We provide pytorch implementations for the paper LIT: bLockwise Intermediate Representation Training for Model Compression
The code was written by Animesh Koratana and Daniel Kang. 

## Getting Started
#### Installing Dependencies
We recommend using virtualenv and building this environment inside of a new environment. The code has been written to work on python3.5. 
```sh
$ pip install -r requirements.txt
```

#### Getting the Data
The data is provided by a third party
[here](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M).
See [here](https://github.com/zonetrooper32/VDCNN) for the original source of
the data.

### Compress the Teacher Model Using LIT
##### Pretrained Models
We provide pretrained models for VDCNN [here](https://drive.google.com/open?id=1visCoagNdVWN_Z9K49T26yzv6Nnw55dH).
The models inside the folder we provide 3 pretrained teacher models in the teachers directory: a vdcnn29-maxpool, vdcnn17-maxpool, and a vdcnn-9-maxpool trained from scratch using regular training methods. 

We additionally provide one student model in the students directory: a vdcnn-9-maxpool trained using the LIT method. The configuration file to reproduce this model is below.

| Model Name                                      | Architecture    | Accuracy |
|-------------------------------------------------|-----------------|----------|
| teachers/vdcnn29.pth                            | vdcnn29-maxpool | 63.03    |
| teachers/vdcnn17.pth                            | vdcnn17-maxpool | 62.90    |
| teachers/vdcnn9.pth                             | vdcnn9-maxpool  | 62.25    |
| students/vdcnn-9-student/lit_vdcnn9_maxpool.pth | vdcnn9-maxpool  | 63.06    |

##### Structure
`trainlit.py` - This is where the main function of the program is. It initializes and loads the student and teacher models, prepares the dataset constructs an object with all the LIT environment variables, and passes it to the LitTrainer, which then trains the models. 

`lit.py` - This file holds the the code to train a student model from a teacher model using LIT

`trainlit.py` - This file holds the code that sets up the appropriate sections, dataset, and copied sections before handing the data to the LIT Trainer to train the student model. This file also has the main function that parses the configuration files passed in.

##### Compression Config Files
The `trainlit.py` file takes in a config file to know all of the needed hyperparameters, architectures, and model paths. Here is an example of the config file you can feed into the script to train a vdcnn-9-maxpool off of a vdcnn-17-maxpool (as done in the paper).

```
#### config.yml ######
models:
  dataset: amazon_review_full
  training_architecture: vdcnn9-maxpool
  training_checkpoint:
  base_architecture: vdcnn17-maxpool
  base_model_path: ./models/teachers/vdcnn17.pth # relative path to the ground truth model

sequence:
  lit:
    epochs: 15
    lr: 0.01
    milestones: [3,6,9,12,15]
  full_model:
    epochs: 10
    lr: 0.001
    milestones: [3, 6, 9]

params:
  logdir: ./runs/lit_vdcnn9-maxpool
  save_directory: ./models/lit_vdcnn9-maxpool
  save_model_name: lit_vdcnn9_maxpool.pth
  evaluate: False
  save_every: 1

hyperparameters:
  alpha: 0.95
  beta: 0.05
  temperature: 6.0
  batch_size: 128
  data_loading_workers: 10
  momentum: 0.9
  weight_decay: 0.0001
  n_random_images: 0
```
To run the program with this configuration, save the following text to a file called `config.yml`. Then run the following command 
```sh
$ python trainlit.py --config config.yml
```

Get LIT!

## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0](http://pytorch.org/)


## Usage

### 1. Cloning the repository
```bash
$ git clone https://github.com/stanford-futuredata/lit-code.git
$ cd lit-code/stargan
```

### 2. Downloading the dataset
To download the CelebA dataset:
```bash
$ bash download.sh celeba
```
### 3. Downloading teacher models
Download and unzip the folder with pretrained teacher networks from
[here](https://drive.google.com/open?id=17C6IRZsOjHxp2d5YGjVXaMzFjQQn_zTG) into the `stargan` directory.

### 3. Training
To train StarGAN student on CelebA from the teacher models, run the training script below.

```bash
$ python main.py  --mode train --dataset CelebA --image_size 128 --c_dim 5 --sample_dir stargan_celeba/samples \
                  --log_dir stargan_celeba/logs --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                   --model_save_dir stargan_models --g_repeat_num {NUM_RESIDUAL_LAYERS_IN_STUDENT}
```

Replace `NUM_RESIDUAL_LAYERS_IN_STUDENT` with the desired number. We used 2 in
the paper.

### 4. Testing

To test StarGAN on CelebA:

```bash
$ python main.py  --mode test --dataset CelebA --image_size 128 --c_dim 5 --sample_dir stargan_celeba/samples \
                  --log_dir stargan_celeba/logs --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                   --model_save_dir stargan_models --g_repeat_num {NUM_RESIDUAL_LAYERS_IN_STUDENT}
```


## Paper & Acknowledgements
The original StarGAN Paper can be found here:
[StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)

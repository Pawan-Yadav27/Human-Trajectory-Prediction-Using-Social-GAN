# Social GAN

This is the code for the paper

**<a href="https://arxiv.org/abs/1803.10892">Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks</a>**
<br>
<a href="http://web.stanford.edu/~agrim/">Agrim Gupta</a>,
<a href="http://cs.stanford.edu/people/jcjohns/">Justin Johnson</a>,
<a href="http://vision.stanford.edu/feifeili/">Fei-Fei Li</a>,
<a href="http://cvgl.stanford.edu/silvio/">Silvio Savarese</a>,
<a href="http://web.stanford.edu/~alahi/">Alexandre Alahi</a>
<br>
Presented at [CVPR 2018](http://cvpr2018.thecvf.com/)


## Model
Our model consists of three key components: Generator (G), Pooling Module (PM) and Discriminator (D). G is based on encoder-decoder framework where we link the hidden states of encoder and decoder via PM. G takes as input trajectories of all people involved in a scene and outputs corresponding predicted trajectories. D inputs the entire sequence comprising both input trajectory and future prediction and classifies them as “real/fake”.

<div align='center'>
  <img src='images/model.png' width='1000px'>
</div>

## Setup
All code was developed and tested on window 10 with Python 3.9.5 and PyTorch 1.9 (cuda 11.1).
command to install torch dependencies. 
```bash
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

other dependencies are attrdict,numpy,Pillow and six.

You can train your own model by following these instructions:

Step 1: Preparing Data

Run the following script to download the dataset:

```bash
bash scripts/download_data.sh
```

This will create the directory datasets/<dataset_name> with train/ val/ and test/ splits. All the datasets are pre-processed to be in world coordinates i.e. in meters. We support five datasets ETH, ZARA1, ZARA2, HOTEL and UNIV. We use leave-one-out approach, train on 4 sets and test on the remaining set. We observe the trajectory for 8 times steps (3.2 seconds) and show prediction results for 8 (3.2 seconds) and 12 (4.8 seconds) time steps.

## Training the Model

Now you can train a new model by running the script:

```bash
python scripts/train.py
```

python scripts/train.py

By default this will train a model on Zara1, periodically saving checkpoint files checkpoint_with_model.pt and checkpoint_no_model.pt to the current working directory. The training script has a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings:

IF you want to evaluate on pretrained model then refer to this link-
```bash
https://github.com/agrimgupta92/sgan
```



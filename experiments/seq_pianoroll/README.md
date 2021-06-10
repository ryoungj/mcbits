# Lossless sequence compression on pianoroll datasets

Here we include the code for lossless sequential data compression on 4 pianoroll datasets: Nottingham, JSB, MuseData, and Piano-midi.de.

The experimental setup is adapted from the [Filtering variational objectives](https://arxiv.org/pdf/1705.09279.pdf) (FIVO). We use the [FIVO 
implementation](https://github.com/dieterichlawson/models/tree/master/research/fivo) (commit hash `8652bb3`) and copy it in [`fivo/`](./fivo) for simplicity. 

## Usage

The reproduction pipeline consists of:
- building a Python 2 conda environment
- downloading and preprocessing the datasets
- training variational RNN models on the (training) datasets
- compressing the (test) datasets with the trained models

### Build a Python 2 conda environment

To meet the dependency requirements of the FIVO repo, we recommend to 
create a new conda environment with Python 2 and TensorFlow.
```
conda env create -f fivo_environment.yml
conda activate fivo
cd ../..
pip install -e .
```

### Download and preprocess the data 

After downloading the pianoroll datasets, we truncate the sequences to a maximum  length of 100 time steps as discussed in our paper. 
Note that the due to the truncation, we expect the outputs slightly different from the original FIVO paper.
```
export FIVO_CODEREPO=./fivo
export PIANOROLL_DATA=./datasets
export PIANOROLL_CKPT=./checkpoints
bash $PIANOROLL_DATA/create_pianorolls.sh $PIANOROLL_DATA $FIVO_CODEREPO
```

### Training VRNN models
We train 3 models with ELBO, IWAE, and FIVO bounds on each dataset repectively and 4 particles are used. 
We provide the script for training the VRNN models with which you can simply run
``` 
bash train.sh 
```
Note that the numbers of updates are obtained by early stopping on a validation set. We also provide the pretrained checkpoints 
in [checkpoints/](./checkpoints).
For more information about the API of the training routine, please refer to the original FIVO repo.


### Evaluating trained models

Then you can evaluate the trained model on the test set in each setup with the provided script:
```
bash evaluate.sh
```


### Compressing with trained models
To compress with trained models, a typical command is:
```
python compress_seq.py --config $PIANOROLL_DIR/compress_trunc_pianoroll.yml [ARGUMENT_LIST] 
```
The basic compressing arguments are specified in the config file [`configs/compress_trunc_pianoroll.yml`](./configs/compress_trunc_pianoroll.yml) 
and can be overridden by the `[ARGUMENT_LIST]` (see [`mcbits/argsparser.py`](../../mcbits/argsparser.py) for the argument list). 
The key arguments for compressing include:
- `--dataset`: the dataset used for compression 
- `--num_compress`: the maximum number of sequences to compress (set to as large as `10000` to compress the whole test set)
- `--decode_check`: whether run decoding and check the correctness of decoded results
- `--logdir`: the model checkpoint used for compression
- `--latent_size`, `--bound`: the latent dimension and the variational bound specified at the training time 
- `--coder`: the coder to use for compression (should fixed to `SMCBitsBackCoder` (BB-SMC) for this experiment)
- `--num_particles`: number of particles for compression
- `--resample`: whether apply resampling for BB-SMC. If `False`, BB-SMC reduces to BB-IS
- `--adaptive`: whether apply *adaptive* resampling for BB-SMC
- `--lprec`, `--bprec`: the precisions for the rANS stack
- `--log_num_bucket`, `--prior_mprec`, `--prop_mprec`, `--cond_mprec`: the precisions for discretizing the latent and observation distributions

We also provide a script for simply run compression with models trained in each setup:
```
bash compress.sh
```
The script compresses `--num_compress=50` sequences for each setup and we provide results below for reference. **Net** bitrates 
(bits/sym) are compared.

| Dataset | Musedata | Nottingham |  JSB | Piano. |
| --- | :---: | :---: | :---: | :---: |
| BB-ELBO       | 9.41 | 6.05 | 12.54 | 10.57 | 
| BB-IS (4)     | 9.43 | 4.96 | 11.90 | 10.52 |
| BB-SMC (4)    | 8.66 | 4.87 | 10.96 | 10.37 |
| Savings       | 8.0%  | 19.5% | 12.6% | 1.9% |

Note that since the number of sequences compressed is relatively small, there maybe a large gap between the total bitrate 
and the net bitrate which decreases as `--num_compress` increases. Futhermore, the gap of BB-IS and BB-SMC is larger than 
that of BB-ELBO which can be addressed by the coupling technique as discussed in our paper, but we haven't implemented it 
for sequential coders. 

# The experiemental setup specified by the dataset and the variational bound used for training the model
# Change these to the set of setups that you want to evaluate
bounds=("elbo" "iwae" "fivo")
datasets=("trunc_jsb" "trunc_musedata" "trunc_nottingham" "trunc_piano-midi.de")

for data in ${datasets[@]}; do
  for bound in ${bounds[@]}; do

    # specify the latent size
    if [[ "${data}" == *"jsb"* ]]; then
      latent_size=32
    elif [[ "${data}" == *"musedata"* ]]; then
      latent_size=256
    elif [[ "${data}" == *"nottingham"* ]]; then
      latent_size=64
    elif [[ "${data}" == *"piano-midi.de"* ]]; then
      latent_size=64
    else
      echo "Dataset ${data} not invalid!!!"
    fi

    # specify the coder used for compression corresponding to the variational bound used for training the models
    if [[ "${bound}" == "elbo" ]]; then
      # BB-ELBO
      num_particles=1
      resample=False
      adaptive=False
    elif [ "${bound}" == "iwae" ]; then
      # BB-IS
      num_particles=4
      resample=False
      adaptive=False
    elif [ "${bound}" == "fivo" ]; then
      # BB-SMC with adaptive resampling
      num_particles=4
      resample=True
      adaptive=True
    else
      echo "Bound ${bound} not invalid!!!"
    fi

    # we found using higher precisions for nottingham leads to better net bitrates
    # increasing precisions will also increase total bitrates
    if [[ "${data}" == *"nottingham"* ]]; then
      overide_params="--lprec=64 --bprec=54 --log_num_bucket 24 --prop_mprec 40 --prior_mprec 40 --cond_mprec 24"
    else
      overide_params=""
    fi

    # run compression
    python compress_seq.py \
      --config configs/compress_trunc_pianoroll.yml \
      --dataset ${data} \
      --dataset_path="$PIANOROLL_DATA/$data.pkl" \
      --logdir=$PIANOROLL_CKPT/$data/$bound \
      --latent_size=$latent_size \
      --bound=$bound \
      --num_particles=$num_particles \
      --resample=$resample \
      --adaptive=$adaptive \
      $overide_params

  done
done

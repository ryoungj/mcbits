# The experiemental setup specified by the dataset and the variational bound used for training the model
# Change these to the set of setups that you want to evaluate
bounds=("elbo" "iwae" "fivo")
datasets=("trunc_jsb" "trunc_musedata" "trunc_nottingham" "trunc_piano-midi.de")

for data in ${datasets[@]}; do
  for bound in ${bounds[@]}; do

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

    if [[ "${bound}" == "elbo" ]]; then
      num_samples=1
    elif [ "${bound}" == "iwae" ]; then
      num_samples=4
    elif [ "${bound}" == "fivo" ]; then
      num_samples=4
    else
      echo "Bound ${bound} not invalid!!!"
    fi

    echo "Evaluating $data with $bound"

    echo "Evaluation with $num_samples particles"
    python $FIVO_CODEREPO/run_fivo.py --mode=eval --split=test \
      --logdir=$PIANOROLL_CKPT/$data/$bound \
      --latent_size=$latent_size \
      --model=vrnn \
      --batch_size=4 --num_samples=$num_samples \
      --dataset_path="$PIANOROLL_DATA/$data.pkl" \
      --dataset_type="pianoroll"

    echo "Estimated likelihood estimation with 256 particles"
    python $FIVO_CODEREPO/run_fivo.py --mode=eval --split=test \
      --logdir=$PIANOROLL_CKPT/$data/$bound \
      --latent_size=$latent_size \
      --model=vrnn \
      --batch_size=4 --num_samples=256 \
      --dataset_path="$PIANOROLL_DATA/$data.pkl" \
      --dataset_type="pianoroll"
  done
done

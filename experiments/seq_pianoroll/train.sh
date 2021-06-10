bounds=("elbo" "iwae" "fivo")
datasets=("trunc_jsb" "trunc_musedata" "trunc_nottingham" "trunc_piano-midi.de")
latent_sizes=(32 256 64 64)
# the training updates are selected by the best models on the evaluation set for each setup
updates=(130000 140000 280000 88000 92000 104000 380000 630000 740000 106000 111000 122000)

for ((i = 0; i < 12; i++)); do
  bound=${bounds[i % 3]}
  data=${datasets[i / 3 % 3]}
  latent_size=${latent_sizes[i / 3 % 3]}
  update=${updates[i]}

  echo "Following the results for $bound and $data; "

  python $FIVO_CODEREPO/run_fivo.py --mode=train --bound=$bound \
    --logdir=$PIANOROLL_CKPT/$data/$bound \
    --dataset_path="$PIANOROLL_DATA/$data.pkl" \
    --latent_size=$latent_size \
    --batch_size=4 \
    --num_samples=4 \
    --model=vrnn \
    --summarize_every=50 \
    --learning_rate=0.00003 \
    --max_steps=$update \
    --dataset_type="pianoroll"
done

if [ -z "$1" ]
  then
    echo "Error, must provide a directory to download the files to."
    exit
fi

mkdir -p $1
echo "Downloading datasets into $1"

curl -s "http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle" > $1/piano-midi.de.pkl
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle" > $1/nottingham.pkl
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle" > $1/musedata.pkl
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle" > $1/jsb.pkl

for data in "nottingham" "jsb" "musedata" "piano-midi.de"; do
  echo "Truncating $data..."
  python `dirname $0`/create_truncated_pianorolls.py --in_file=$1/$data.pkl --out_file=$1/trunc_$data.pkl --seq_len=100
  echo "Processing $data..."
  python $2/fivo/data/calculate_pianoroll_mean.py --in_file=$1/$data.pkl
  python $2/fivo/data/calculate_pianoroll_mean.py --in_file=$1/trunc_$data.pkl
  echo "Preprocessed $data!"
done


set -euo pipefail
set -x
BASE_DIR=$(pwd)

# STARGO
# python $BASE_DIR/bin/evaluate.py -c $BASE_DIR/configs/ordered_encdec_medium.toml -d $BASE_DIR/datasets/pfresgo -s bp -r 2020 -b 16 -w
python $BASE_DIR/bin/evaluate.py -c $BASE_DIR/configs/ordered_encdec_medium.toml -d $BASE_DIR/datasets/pfresgo -s cc -r 2020 -b 32 -w
# python $BASE_DIR/bin/evaluate.py -c $BASE_DIR/configs/ordered_encdec_medium.toml -d $BASE_DIR/datasets/pfresgo -s mf -r 2020 -b 32 -w

# DeepGOZero zero-shot
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero.toml -d $BASE_DIR/datasets/deepgozero -s cc -r 2020 -b 32
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero.toml -d $BASE_DIR/datasets/deepgozero -s mf -r 2020 -b 16
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero.toml -d $BASE_DIR/datasets/deepgozero -s bp -r 2020 -b 16

# DeepGOZero normal
# python $BASE_DIR/bin/evaluate.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_textual.toml -d $BASE_DIR/datasets/deepgozero -s bp -r 2020 -w --eval-mode zero

# Textual-only ablation
#python $BASE_DIR/bin/evaluate.py -c $BASE_DIR/configs/ordered_encdec_medium_textual.toml -s cc -r 2020 -w


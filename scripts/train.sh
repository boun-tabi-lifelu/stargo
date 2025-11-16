set -euo pipefail
BASE_DIR=$(pwd)

# STARGO
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium.toml -d $BASE_DIR/datasets/pfresgo -s bp -r 2020 -b 16
python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium.toml -d $BASE_DIR/datasets/pfresgo -s cc -r 2020 -b 32
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium.toml -d $BASE_DIR/datasets/pfresgo -s mf -r 2020 -b 32

# DeepGOZero zero-shot
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero.toml -d $BASE_DIR/datasets/deepgozero -s cc -r 2020 -b 32
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero.toml -d $BASE_DIR/datasets/deepgozero -s mf -r 2020 -b 16
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero.toml -d $BASE_DIR/datasets/deepgozero -s bp -r 2020 -b 16

# DeepGOZero zero-shot PU-ranking loss
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_pu_ranking.toml -d $BASE_DIR/datasets/deepgozero -s cc -r 2020 -b 32
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_pu_ranking.toml -d $BASE_DIR/datasets/deepgozero -s mf -r 2020 -b 32
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_pu_ranking.toml -d $BASE_DIR/datasets/deepgozero -s bp -r 2020 -b 16

# DeepGOZero zero-shot with textual embeddings
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_textual.toml -d $BASE_DIR/datasets/deepgozero -s cc -r 2020 -b 32
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_textual.toml -d $BASE_DIR/datasets/deepgozero -s mf -r 2020 -b 16
#python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_textual.toml -d $BASE_DIR/datasets/deepgozero -s bp -r 2020 -b 8

# DeepGOZero zero-shot with anc2vec embeddings
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_structural.toml -d $BASE_DIR/datasets/deepgozero -s cc -r 2020 -b 32
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_structural.toml -d $BASE_DIR/datasets/deepgozero -s mf -r 2020 -b 16
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_actual_zero_structural.toml -d $BASE_DIR/datasets/deepgozero -s bp -r 2020 -b 8

# Textual-only ablation for PFresGO
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_textual.toml -s bp -r 2020 -b 16
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_textual.toml -s mf -r 2020 -b 32
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_textual.toml -s cc -r 2020 -b 32

# PU ranking ablation
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_pu_ranking.toml -s bp -r 2020 -b 16
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_pu_ranking.toml -s mf -r 2020 -b 32
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_pu_ranking.toml -s cc -r 2020 -b 32

# PU ranking with priors
# python $BASE_DIR/bin/train.py -c $BASE_DIR/configs/ordered_encdec_medium_pu_ranking_priors.toml -s cc -r 2020 -b 32


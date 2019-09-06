# Extract program traces from the Kaggle database
# and remove any programs that we determine are 
# pseudo-duplicates

DATA_DIR=../../data/meta-kaggle
DATABASE=$DATA_DIR/database.sqlite
OUTPUT=$DATA_DIR/extracted-traces.pkl
CLEAN_OUTPUT=$DATA_DIR/clean-traces.pkl
LANGUAGE=Python

./traces.py $DATABASE --output $OUTPUT --language $LANGUAGE
./remove_pseudo_duplicates.py $OUTPUT $CLEAN_OUTPUT
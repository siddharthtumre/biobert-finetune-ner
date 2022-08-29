#!/bin/bash
ENTITIES="JNLPBA"
MAX_LENGTH=128

for ENTITY in $ENTITIES
do
	echo "***** " $ENTITY " Preprocessing Start *****"
	DATA_DIR=./datasets/NER/$ENTITY

	# Preprocess for BERT-based models
	python scripts/preprocess.py $DATA_DIR/train.txt.tmp bert-base-cased $MAX_LENGTH > $DATA_DIR/train.txt
	python scripts/preprocess.py $DATA_DIR/valid.txt.tmp bert-base-cased $MAX_LENGTH > $DATA_DIR/valid.txt
	python scripts/preprocess.py $DATA_DIR/test.txt.tmp bert-base-cased $MAX_LENGTH > $DATA_DIR/test.txt
	cat $DATA_DIR/train.txt $DATA_DIR/valid.txt $DATA_DIR/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_DIR/labels.txt
	echo "***** " $ENTITY " Preprocessing Done *****"
done

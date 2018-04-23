# Usage: bash ./data/tools/enwiki/process.sh

ENWIKI_DIR='data/preprocessed/enwiki'

mkdir -p $ENWIKI_DIR/enwiki_processed

for i in {00..11};
do
    echo "enwiki preprocess ${i}"
    python data/tools/enwiki/wiki-corpus-prepare.py $ENWIKI_DIR/enwiki_extracted/wiki_$i $ENWIKI_DIR/enwiki_processed/wiki_$i &
done

echo "done"

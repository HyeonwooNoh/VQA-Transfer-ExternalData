#!/bin/bash
mkdir BookCorpus
cd BookCorpus
wget --user mbweb --password Tiwp49mS3Ts http://www.cs.toronto.edu/~mbweb/data/books_in_sentences.tar
tar -zxvf books_in_sentences.tar
rm books_in_sentences.tar

split -n l/5 books_large_p1.txt --additional-suffix=.txt
split -n l/5 books_large_p2.txt --additional-suffix=.txt

mv books_large_p1.txt books_large_p1.txt.backup
mv books_large_p2.txt books_large_p2.txt.backup

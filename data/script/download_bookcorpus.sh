#!/bin/bash
mkdir BookCorpus
cd BookCorpus
wget --user mbweb --password Tiwp49mS3Ts http://www.cs.toronto.edu/~mbweb/data/books_in_sentences.tar
unzip books_in_sentences.tar
rm books_in_sentences.tar

# Data
Directory containing datasets

### Preparation
1. Download [VisualGenome version 1.4](http://visualgenome.org/api/v0/api_home.html)
1. Download [GloVe word vector](https://nlp.stanford.edu/projects/glove/)
You can use the following command
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip glove.6B.zip
```

### Structure (After downloading)
VisualGenome/VG\_100K/\*.jpg
VisualGenome/annotations/objects.json
VisualGenome/annotations/attributes.json
VisualGenome/annotations/relationships.json
VisualGenome/annotations/region\_descriptions.json
VisualGenome/annotations/region\_graphs.json
VisualGenome/annotations/scene\_graphs.json
VisualGenome/annotations/object\_alias.txt
VisualGenome/annotations/relationship\_alias.txt

GloVe/glove.6B.50d.txt
GloVe/glove.6B.100d.txt
GloVe/glove.6B.200d.txt
GloVe/glove.6B.300d.txt

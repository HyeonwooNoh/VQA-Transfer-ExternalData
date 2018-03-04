mkdir VQA_v2
cd VQA_v2

mkdir annotations
cd annotations
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
rm v2_Annotations_Train_mscoco.zip
rm v2_Annotations_Val_mscoco.zip
cd ..

mkdir questions
cd questions
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
unzip v2_Questions_Test_mscoco.zip
cd ..

mkdir images
cd images
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip
unzip train2014.zip
unzip val2014.zip
unzip test2015.zip
rm train2014.zip
rm val2014.zip
rm test2015.zip
cd ..

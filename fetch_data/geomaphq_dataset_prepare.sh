#mkdir geomap-hq-dataset
#
#unzip data256x256.zip -d geomap-hq-dataset/
#
# Reindex
for i in `echo {00001..05507}`
do
    mv 'geomap-hq-dataset/data256x256/'$i'.jpg' 'geomap-hq-dataset/data256x256/'$[10#$i - 1]'.jpg'
done

ls geomap-hq-dataset//data256x256/ | shuf > geomap-hq-dataset//temp_all_shuffled.flist

head -n 500 geomap-hq-dataset//temp_all_shuffled.flist > geomap-hq-dataset//val_shuffled.flist
tail -n +501 geomap-hq-dataset//temp_all_shuffled.flist | head -n 500 > geomap-hq-dataset//visual_test_shuffled.flist
tail -n +1001 geomap-hq-dataset//temp_all_shuffled.flist > geomap-hq-dataset//train_shuffled.flist

mkdir geomap-hq-dataset//train_256/
mkdir geomap-hq-dataset//val_source_256/
mkdir geomap-hq-dataset//visual_test_source_256/

cat geomap-hq-dataset//train_shuffled.flist | xargs -I {} mv geomap-hq-dataset//data256x256/{} geomap-hq-dataset//train_256/
cat geomap-hq-dataset//val_shuffled.flist | xargs -I {} mv geomap-hq-dataset//data256x256/{} geomap-hq-dataset//val_source_256/
cat geomap-hq-dataset//visual_test_shuffled.flist | xargs -I {} mv geomap-hq-dataset//data256x256/{} geomap-hq-dataset//visual_test_source_256/

# create location config celeba.yaml
PWD=$(pwd)
DATASET=${PWD}/geomap-hq-dataset
GEOMAP=${PWD}/configs/training/location/geomap.yaml

touch $GEOMAP
echo "# @package _group_" >> $GEOMAP
echo "data_root_dir: ${DATASET}/" >> $GEOMAP
echo "out_root_dir: ${PWD}/experiments/" >> $GEOMAP
echo "tb_dir: ${PWD}/tb_logs/" >> $GEOMAP
echo "pretrained_models: ${PWD}/" >> $GEOMAP

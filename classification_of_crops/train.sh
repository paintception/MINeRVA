CUDA=$1
LR=$2

PROJECT=/home/nbanar/pycharmProjects/MINeRVA/classification_of_crops
DATA=/home/nbanar/pycharmProjects/datasets/minerva_classification
MODELS=/home/nbanar/pycharmProjects/art_detector/classification_of_crops/ECCVModels/

for SET in top5 top10 top20 hypernym all
do
    for NET in ResNet V3 VGG19
    do
        echo "Looping ... i is set to ${NET} and ${SET} and ${LR}"
        CUDA_VISIBLE_DEVICES=${CUDA} python ${PROJECT}/train_and_predict.py -data ${DATA}/${SET}/ -model_path ${MODELS} -net ${NET} -save ${PROJECT}/results/${SET}/${NET}/${LR}/ -lr ${LR}
    done
done

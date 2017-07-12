NB_ROC=50

make build

for i in $(seq 1 $NB_ROC)
do
    TF_CPP_MIN_LOG_LEVEL=1 python main.py 
done

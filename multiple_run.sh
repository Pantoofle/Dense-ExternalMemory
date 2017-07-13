NB_ROC=2

make build

for i in $(seq 1 $NB_ROC)
do
    TF_CPP_MIN_LOG_LEVEL=1 python main.py rocs/roc$i.html
done

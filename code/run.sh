rm out
source ~/torch_env/bin/activate
python3 -u train.py  --lr=0.0001 --num_epochs=10 --batch_size=64  --weight_decay=0.0001

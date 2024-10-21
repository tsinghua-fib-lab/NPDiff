#********* periodic dynamics *********#
# MobileBJ
python main.py --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "CSDI" --target_dim 672 --history_len 12 --predict_len 1 --data_name "MobileBJ" 
python main.py --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "STID" --target_dim 672 --history_len 12 --predict_len 1 --data_name "MobileBJ" 


# MobileNJ
python main.py --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "CSDI" --target_dim 560 --history_len 12 --predict_len 1 --data_name "MobileNJ"
python main.py --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "STID" --target_dim 560 --history_len 12 --predict_len 1 --data_name "MobileNJ"


# MobileSH14
python main.py --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "CSDI" --target_dim 896 --history_len 12 --predict_len 1 --data_name "MobileSH14"
python main.py --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "STID" --target_dim 896 --history_len 12 --predict_len 1 --data_name "MobileSH14"


# MobileSH16
python main.py --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "CSDI" --target_dim 400 --history_len 12 --predict_len 1 --data_name "MobileSH16"
python main.py --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "STID" --target_dim 400 --history_len 12 --predict_len 1 --data_name "MobileSH16"





#********* local dynamics *********#
# MobileBJ
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "CSDI" --target_dim 672 --history_len 12 --predict_len 1 --data_name "MobileBJ" 
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "ConvLSTM" --target_dim 672 --history_len 12 --predict_len 1 --data_name "MobileBJ" 
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "STID" --target_dim 672 --history_len 12 --predict_len 1 --data_name "MobileBJ" 


# MobileNJ
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "CSDI" --target_dim 560 --history_len 12 --predict_len 1 --data_name "MobileNJ"
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "ConvLSTM" --target_dim 560 --history_len 12 --predict_len 1 --data_name "MobileNJ"
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "STID" --target_dim 560 --history_len 12 --predict_len 1 --data_name "MobileNJ"


# MobileSH14
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "CSDI" --target_dim 896 --history_len 12 --predict_len 1 --data_name "MobileSH14"
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "ConvLSTM" --target_dim 896 --history_len 12 --predict_len 1 --data_name "MobileSH14"
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "STID" --target_dim 896 --history_len 12 --predict_len 1 --data_name "MobileSH14"


# MobileSH16
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "CSDI" --target_dim 400 --history_len 12 --predict_len 1 --data_name "MobileSH16"
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "ConvLSTM" --target_dim 400 --history_len 12 --predict_len 1 --data_name "MobileSH16"
python main.py --local_dynamics --batch_size 8 --device "cuda:2" --Num_Comp 0 --Lambda 0.5 --model "STID" --target_dim 400 --history_len 12 --predict_len 1 --data_name "MobileSH16"

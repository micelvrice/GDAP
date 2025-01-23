## paddle版本
-  paddlepaddle-gpu 3.0.0b2
-  paddlenlp 3.0.0b3

## 微调脚本
python -u -m paddle.distributed.launch --gpus "0,1,2,3" run_finetune.py lora_argument.json


All experiments were conducted using PyTorch 2.5.1 and Python 3.12 with CUDA
12.4. The model was trained on a single NVIDIA RTX 3090 GPU (24 GB) equipped
with an Intel Xeon Gold 6330 CPU. 
2. run python -m prompter.main --config cell_config.py --model-ema --output_dir light --eval --resume prompter/checkpoint/light/best.pth to see the evaluation result
3. run python -m prompter.main --config cell_config.py --model-ema --output_dir light --epoch 130 to train the model 
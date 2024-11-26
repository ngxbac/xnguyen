torchrun --nproc-per-node=4 --master-port=2411 test_train.py    --output_dir ./logs/baseline/ \
                                                                    --batch_size_per_gpu 32
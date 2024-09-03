# FSDNet

## Revisiting Shadow Detection: A New Benchmark Dataset for Complex World

by Xiaowei Hu, Tianyu Wang, Chi-Wing Fu, Yitong Jiang, Qiong Wang, and Pheng-Ann Heng.

## Fast Shadow Detection Network

This implementation is written by Xiaowei Hu at the Chinese University of Hong Kong.

***

## Citation
```
@article{hu2021revisiting,            
    title={Revisiting Shadow Detection: A New Benchmark Dataset for Complex World},             
    author={Hu, Xiaowei and Wang, Tianyu and Fu, Chi-Wing and Jiang, Yitong and Wang, Qiong and Heng, Pheng-Ann},                  
    journal={IEEE Transactions on Image Processing},               
    volume={30},                   
    pages={1925--1934},                   
    year={2021}
}
```

## Requirements

* PyTorch1.3.0
* Python3.6
* Cupy [[Installation Guide](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy)]

  
## Train and Test

1. Clone this repository:          
   ```shell
   git clone https://github.com/xw-hu/FSDNet.git
   ```
   
2. Train:
   ```shell
   python3 train.py    
   ```
   
3. Test:
   ```shell
   python3 infer.py
   ```
   
## CUHK-Shadow Dataset
Please find the dataset at [https://github.com/xw-hu/CUHK-Shadow#cuhk-shadow-dateset](https://github.com/xw-hu/CUHK-Shadow#cuhk-shadow-dateset).

## Evaluation Function 
Please find the evaluation function at [https://github.com/xw-hu/CUHK-Shadow#cuhk-shadow-evaluation](https://github.com/xw-hu/CUHK-Shadow#cuhk-shadow-evaluation).

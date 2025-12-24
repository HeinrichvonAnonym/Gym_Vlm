#
implemente vlm based on isaacgym env
## TODO:
1. configuration of multi franka initialization (done)
2. configutation of multi camera initialization (done)
3. osc control directely based on isaacgym apis (done)
4. PTP command (done)
5. rgbd datapipeline (done)
6. pcl datapipeline (done)
7. segmentation of rgbd info (doing)
8. vlm agent setting (doing)
9. fine modeling of scenario (doing)
10. fine rendering of scenario (doing)

## install

### prepare
1. install cuda recommand  11.7
2. install torch recommand  1.13.1 + cuda 11.7 with python 3.8
3. install isaacgym
4. install sam and download pretrained model
5. install groundingdino and download pretrained model
6. download pretrained BERT model

### install packages
``` bash
    pip install requirements.txt
```

## run
python main.py

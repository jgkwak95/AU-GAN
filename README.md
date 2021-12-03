# Adverse Weather Image Translation with Asymmetric and Uncertainty-aware GAN (AU-GAN)
Official Tensorflow implementation of [Adverse Weather Image Translation with Asymmetric and Uncertainty-aware GAN](https://www.bmvc2021-virtualconference.com/assets/papers/1443.pdf) (AU-GAN)\
Jeong-gi Kwak, Youngsaeng Jin, Yuanming Li, Dongsik Yoon, Donghyeon Kim and Hanseok Ko </br>
*British Machine Vision Conference (BMVC), 2021*
</br>

## Results
### Night &rarr; Day ([BDD100K](https://bdd-data.berkeley.edu/))
<img src="./assets/augan_bdd.png" width="800">

### Rainy night &rarr; Day ([Alderdey](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395))
<img src="./assets/augan_alderley.png" width="800">
</br>


## **Envs**

```bash

git clone https://github.com/jgkwak95/AU-GAN.git
cd AU-GAN

# Create virtual environment
conda create -y --name augan python=3.6.7
conda activate augan

conda install tensorflow-gpu==1.14.0   # Tensorflow 1.14
pip install --no-cache-dir -r requirements.txt

```

## **Preparing datasets**

**Night &rarr; Day** </br>
[Berkeley DeepDrive dataset](https://bdd-data.berkeley.edu/) contains 100,000 high resolution images of the urban roads for autonomous driving.</br></br>
**Rainy night &rarr; Day** </br>
[Alderley dataset](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395) consists of images of two domains,
rainy night and daytime. It was collected while driving the same route in each weather environment.</br>
</br>
Please download datasets and then construct them following [ForkGAN](https://github.com/zhengziqiang/ForkGAN)

## Training

```bash

# Alderley (256x256)
python main_uncer.py --dataset_dir alderley
                     --phase train
                     --experiment_name alderley_exp
                     --batch_size 8 
                     --load_size 286 
                     --fine_size 256 
                     --use_uncertainty True

```

```bash

# BDD100k (512x512)
python main_uncer.py --dataset_dir bdd100k 
                     --phase train
                     --experiment_name bdd_exp
                     --batch_size 4 
                     --load_size 572 
                     --fine_size 512 
                     --use_uncertainty True

```

## Test

```bash

# Alderley (256x256)
python main_uncer.py --dataset_dir alderley
                     --phase test
                     --experiment_name alderley_exp
                     --batch_size 1 
                     --load_size 286 
                     --fine_size 256 
                    
```

```bash

# BDD100k (512x512)
python main_uncer.py --dataset_dir bdd100k
                     --phase test
                     --experiment_name bdd_exp
                     --batch_size 1 
                     --load_size 572 
                     --fine_size 512 
                    

```


## **Citation**
If our code is helpful your research, please cite our paper:
```
@InProceedings{kwak_adverse_2021},
  author = {Jeong-gi, Kwak and Youngsaeng, Jin and Yuanming, Li and Dongsik, Yoon and  Donghyeon, Kim and Hanseok, Ko},
  title = {Adverse Weather Image Translation with Asymmetric and Uncertainty-aware GAN},
  booktitle = {British Conference of Computer Vision (BMVC)},
  month = {November},
  year = {2021}
}
```
## Acknowledgments
Our code is bulided upon the [ForkGAN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480154.pdf) implementation.

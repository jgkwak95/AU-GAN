# Adverse Weather Image Translation with Asymmetric and Uncertainty-aware GAN (AU-GAN)
Official Tensorflow implementation of [Adverse Weather Image Translation with Asymmetric and Uncertainty-aware GAN](https://www.bmvc2021-virtualconference.com/assets/papers/1443.pdf) (AU-GAN)\
Jeong-gi Kwak, Youngsaeng Jin, Yuanming Li, Dongsik Yoon, Donghyeon Kim and Hanseok Ko </br>
*British Machine Vision Conference (BMVC), 2021*
</br>

## Intro 

### Night &rarr; Day ([BDD100K](https://bdd-data.berkeley.edu/))
<img src="./assets/augan_bdd.png" width="800">

### Rainy night &rarr; Day ([Alderdey](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395))
<img src="./assets/augan_alderley.png" width="800">
</br>


## Architecture
<img src="./assets/augan_model.png" width="800">
Our generator has asymmetric structure for editing day&rarr;night and night&rarr;day.
Please refer our paper for details

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

# BDD100k (256x256)
python main_uncer.py --dataset_dir bdd100k 
                     --phase train
                     --experiment_name bdd_exp
                     --batch_size 8 
                     --load_size 286
                     --fine_size 256 
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

# BDD100k (256x256)
python main_uncer.py --dataset_dir bdd100k
                     --phase test
                     --experiment_name bdd_exp
                     --batch_size 1 
                     --load_size 286 
                     --fine_size 256 
                    

```
## Additional results
<img src="./assets/augan_result.png" width="800">

More results in [paper](https://www.bmvc2021-virtualconference.com/assets/papers/1443.pdf) and [supplementary]()

## Uncertainty map 
<img src="./assets/augan_uncer.png" width="800">

## **Citation**
If our code is helpful your research, please cite our paper:
```
@article{kwak2021adverse,
  title={Adverse weather image translation with asymmetric and uncertainty-aware GAN},
  author={Kwak, Jeong-gi and Jin, Youngsaeng and Li, Yuanming and Yoon, Dongsik and Kim, Donghyeon and Ko, Hanseok},
  journal={arXiv preprint arXiv:2112.04283},
  year={2021}
}
```
## Acknowledgments
Our code is bulided upon the [ForkGAN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480154.pdf) implementation.

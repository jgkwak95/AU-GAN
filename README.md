# Adverse Weather Image Translation with Asymmetric and Uncertainty-aware GAN (BMVC 2021)
Official code of [Adverse Weather Image Translation with Asymmetric and Uncertainty-aware GAN](https://jgkwak95.github.io/) (AU-GAN)\
Jeong-gi Kwak, Youngsaeng Jin, Yuanming Li, Dongsik Yoon, Donghyeon Kim and Hanseok Ko </br>
*British Machine Vision Conference (BMVC), 2021*
</br>

### Night &arar; day ([BDD100K](https://bdd-data.berkeley.edu/))
<img src="./assets/augan_bdd.png" width="800">

### Rainy night to day ([Alderdey](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395))
<img src="./assets/augan_alderley.png" width="800">
</br>


## Envs

```bash

git clone https://github.com/jgkwak95/AU-GAN.git
cd AU-GAN

# Create virtual environment
conda create -y --name augan python=3.6.7
conda activate augan

conda install tensorflow-gpu==1.14.0   # Tensorflow 1.14
pip install --no-cache-dir -r requirements.txt

```

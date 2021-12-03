import json
import shutil


with open('C:/jg/github_code/ForkGAN/bdd100k/labels/bdd100k_labels_images_train.json') as json_file:
    json_data = json.load(json_file)

for item in json_data:
    item_path = 'C:/jg/github_code/ForkGAN/bdd100k/images/100k/train/'+ item['name']
    print(item['name'])
    if item['attributes']['timeofday'] == 'daytime':
        shutil.copy(item_path, 'C:/jg/github_code/ForkGAN/bdd100k/images/daytime/'+item['name'])

    elif item['attributes']['timeofday'] == 'night':
        shutil.copy(item_path, 'C:/jg/github_code/ForkGAN/bdd100k/images/night/'+item['name'])

    else :
        shutil.copy(item_path, 'C:/jg/github_code/ForkGAN/bdd100k/images/else/' + item['name'])
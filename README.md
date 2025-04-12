# T-CADiff

This is the implementaion of T-CADiff. 

This implementation adopted the code from [improved-diffusion](https://github.com/openai/improved-diffusion).

You can download the pretrained model from [here](https://drive.google.com/file/d/1uYc3rhrXvoTPI8ahat-SM3KSGiLbEFao/view?usp=drive_link). The dataset used in our experiments can be downloaded from [here](https://drive.google.com/drive/folders/1KfFj0R8rqIfktFhuCgftdtsMgBXL-zJ3?usp=sharing). 

To train this model on ISIC 2016 Dataset, use function `isic_dataset_agmentation` from `dataset.py` to augment original ISIC 2016 Dataset. The image size is 256*256. In `main.ipynb`, modify variable `path` to the root directory of the augmented dataset. Once everything is set up, run `main.ipynb`.

To test this model on ISIC 2016 Dataset, place the test dataset of ISIC 2016 Dataset and corresponding csv file (from ISIC 2016 Challenge - Task 3B) in the same folder (which servers as the root directory of test dataset). In `model_eval.ipynb`, modify variable `test_data_path` to the root directory of the augmented dataset. Then run `model_eval.ipynb` for evaluation.

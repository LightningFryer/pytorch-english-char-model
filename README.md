# English Alphabet Recognition Model
This project uses timm's "efficientnet_b0" model to create a model to take an image of a handwritten English Alphabet as the input and predict the character as the output. 

## Tech Stack
- python (pytorch, numpy, pandas, matplotlib, timm)
- image model: efficientnet_b0
- dataset: [English Alphabets by Mohneesh_Sreegirisetty from Kaggle](https://www.kaggle.com/datasets/mohneesh7/english-alphabets)

## Training Loss vs Validation Loss Graph
![image](https://github.com/LightningFryer/pytorch-english-char-model/blob/3a92e3e52d3f8014968586a00fd7e7a55d7d07eb/images/Training_Loss_vs_Validation_Loss.png)

## Test Image vs Model Prediction
![image](https://github.com/LightningFryer/pytorch-english-char-model/blob/3a92e3e52d3f8014968586a00fd7e7a55d7d07eb/images/TestCharImageA.png)
![image](https://github.com/LightningFryer/pytorch-english-char-model/blob/3a92e3e52d3f8014968586a00fd7e7a55d7d07eb/images/TestCharImageM.png)
![image](https://github.com/LightningFryer/pytorch-english-char-model/blob/3a92e3e52d3f8014968586a00fd7e7a55d7d07eb/images/TestCharImageX.png)

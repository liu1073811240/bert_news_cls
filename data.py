from datasets import load_dataset

# _TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1yoH_Aa09ECjjhHMHS8ML_d5-ydDwNETu&export=download"
# _DEV_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1GVslP3rnHPoxHHNeEyYDMjFxZVRiSdhM&export=download"
# _TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1NIF3_OHxYM7WgG8xnBP_X393_sQKK43Z&export=download"

# data = load_dataset(path="sesmew/THUCNewsText", split="train")
data = load_dataset(path="csv", data_files="data/validation.csv", split="train")
print(data)

# for i in data:
#     print(i)




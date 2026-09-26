
import pickle

with open("diabetes_model.pkl", "rb") as f:
    payload = pickle.load(f)

model = payload["model"]

print("Final Selected Algorithm:")
print(type(model).__name__)

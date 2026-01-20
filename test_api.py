import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "texture_mean": 10.38,
    "area_mean": 1220.8,
    "concavity_mean": 0.0869,
    "area_se": 153.4,
    "concavity_worst": 0.2671
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())

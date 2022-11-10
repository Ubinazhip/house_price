import requests
import json

almaty_areas = ['Ауэзовский', 'Бостандыкский', 'Алмалинский', 'Алатауский', 'Медеуский', 'Наурызбайский', 'Турксибский',
                'Жетысуский']
curr_dict = {"rooms": 2,
              "sq_m": 40.0,
              "floor": 1,
              "floors_all": 10,
              "area": "Алмалинский",
              "year": "2010"}

url = "http://localhost:9696/predict"
response = requests.post(url, json=curr_dict)
result = response.json()

print(json.dumps(result))

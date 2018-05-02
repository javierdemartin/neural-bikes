import json
import urllib
import requests

url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/2018-01-02T00%3A00%3A00UTC/fechafin/2018-01-22T18%3A00%3A00UTC/estacion/1082"

querystring = {"api_key":"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqYXZpZXJkZW1hcnRpbkBtZS5jb20iLCJqdGkiOiJjNDNiM2RjOS05ZmE4LTQzZjgtOTE5Yi1jOTk3NzIyNGZkYjQiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTUyNDk5ODU2NywidXNlcklkIjoiYzQzYjNkYzktOWZhOC00M2Y4LTkxOWItYzk5NzcyMjRmZGI0Iiwicm9sZSI6IiJ9.uf_MmYF-0NQQ3syfhc8J5QeppKTsMiWqZaNXvL0gZVI"}

headers = {
    'cache-control': "no-cache"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)

#response = urllib.urlopen(response)

data = json.loads(response.text)

print data
print data['datos']

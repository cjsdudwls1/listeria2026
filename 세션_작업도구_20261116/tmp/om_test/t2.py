import urllib.request, urllib.parse, json
d="temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_mean,wind_speed_10m_max,sunshine_duration,shortwave_radiation_sum,relative_humidity_2m_mean,soil_moisture_0_to_7cm_mean"
q=dict(latitude="37.55,37.57469",longitude="126.98,126.96",start_date="2015-07-01",end_date="2015-07-05",daily=d,models="era5_seamless",timezone="auto")
u="https://archive-api.open-meteo.com/v1/archive?"+urllib.parse.urlencode(q)
try:
    r=urllib.request.urlopen(u,timeout=60); b=r.read(); print(r.status, r.headers.get('X-RateLimit-Cost'))
except urllib.error.HTTPError as e:
    b=e.read(); print("HTTP",e.code)
open("resp2.json","wb").write(b)
for x in json.loads(b): print(x['latitude'],x['longitude'],x['timezone'],json.dumps(x['daily']))

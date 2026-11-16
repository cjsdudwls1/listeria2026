import urllib.request, urllib.parse, json, sys
d="temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_mean,wind_speed_10m_max,sunshine_duration,shortwave_radiation_sum,relative_humidity_2m_mean,soil_moisture_0_to_7cm_mean,soil_temperature_0_to_7cm_mean,soil_temperature_7_to_28cm_mean,soil_temperature_28_to_100cm_mean,wind_direction_10m_dominant,dew_point_2m_mean"
q=dict(latitude="37.55",longitude="126.98",start_date="2015-01-01",end_date="2015-01-03",daily=d,models="era5,era5_land",timezone="auto")
u="https://archive-api.open-meteo.com/v1/archive?"+urllib.parse.urlencode(q)
try:
    r=urllib.request.urlopen(u,timeout=60); b=r.read(); print(r.status, dict(r.headers))
except urllib.error.HTTPError as e:
    b=e.read(); print("HTTP",e.code)
open("resp.json","wb").write(b); print(b.decode()[:6000])

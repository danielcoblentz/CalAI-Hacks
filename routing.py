# libraries
import heapq
import math
import requests
import os
import folium
import json
from datetime import datetime
import time
from collections import defaultdict
import numpy as np
import googlemaps


#API keys
GOOGLE_API_KEY =os.getenv("API_KEY")
FIRMS_API_KEY = os.getenv("FIRMS_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)


# Sacramento
# Berkeley
# Fetching functions to compute risk of each type of hazard (fire, tornado, traffic alerts)
def get_google_traffic_data(start_coords, goal_coords):
   print("getting real-time traffic data from Google Maps...")
   start_lat, start_lon = start_coords
   goal_lat, goal_lon = goal_coords
   url = (
       f"https://maps.googleapis.com/maps/api/directions/json"
       f"?origin={start_lat},{start_lon}&destination={goal_lat},{goal_lon}"
       f"&alternatives=true&departure_time=now&traffic_model=best_guess&key={GOOGLE_API_KEY}")
   try:
       response = requests.get(url, timeout=10)
       response.raise_for_status()
       data = response.json()
       if data.get("status") != "OK":
           print(f" issue with Google API error: {data.get('status')} - {data.get('error_message', '')}")
           return []
       best_route = data["routes"][0] 
       import polyline
       coords = polyline.decode(best_route["overview_polyline"]["points"])
       print(f" Route with {len(coords)} points extracted from Google Maps")
       return coords
   except Exception as e:
       return []
  


#gets fire data from nasa.gov site
def fetch_firms_fires(bbox):
   source = "VIIRS_SNPP_NRT"
   days = 1
   west, south, east, north = bbox
   url = (
       f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
       f"{FIRMS_API_KEY}/{source}/{west},{south},{east},{north}/{days}")
   print("getting fire data from NASA FIRMS..")
   try:
       response = requests.get(url, timeout=10)
       response.raise_for_status()
       lines = response.text.strip().split("\n")[1:]
       data = []
       for line in lines:
           parts = line.split(",")
           if len(parts) >= 2:
               lat, lon = float(parts[0]), float(parts[1])
               data.append({"latitude": lat, "longitude": lon, "sensor": source})
       print(f" Found {len(data)} active fire points")
       return data
   except Exception as e:
       print(f" issue fetching fire data check here: {e}")
       return []


# gets data from weather API
def fetch_tornado_alerts():
   print(" Fetching tornado alerts from NWS...")
   url = "https://api.weather.gov/alerts/active?event=Tornado Warning"
   headers = {"User-Agent": "safe-route-app/1.0"}
   try:
       response = requests.get(url, headers=headers, timeout=10)
       response.raise_for_status()
       data = response.json()
       alerts = []
       for feature in data.get("features", []):
           geometry = feature.get("geometry", {})
           if geometry.get("type") == "Polygon":
               for polygon in geometry.get("coordinates", []):
                   alerts.append(polygon)
       print(f" Found {len(alerts)} tornado warning polygons")
       return alerts
   except Exception as e:
       print(f"⚠️ Error fetching tornado data: {e}")
       return []
  
#Assigns a traffic risk score based on distance between points and time delay, using Google traffic data. Returns a list of scores matching the path.
def compute_traffic_risk(path):


   print(" calculating traffic risk per point..")
   traffic_risks = []


   for i in range(len(path) - 1):
       origin = path[i]
       dest = path[i + 1]


       url = (
           f"https://maps.googleapis.com/maps/api/distancematrix/json"
           f"?origins={origin[0]},{origin[1]}&destinations={dest[0]},{dest[1]}"
           f"&departure_time=now&traffic_model=best_guess&key={GOOGLE_API_KEY}")
       try:
           res = requests.get(url, timeout=5)
           res.raise_for_status()
           data = res.json()


           elements = data.get("rows", [{}])[0].get("elements", [{}])[0]
           normal = elements.get("duration", {}).get("value", 0)  #  seconds
           traffic = elements.get("duration_in_traffic", {}).get("value", 0)


           if normal and traffic:
               delay_ratio = traffic / normal
               if delay_ratio >= 2.0:
                   score = 0.6
               elif delay_ratio >= 1.5:
                   score = 0.4
               elif delay_ratio >= 1.2:
                   score = 0.2
               else:
                   score = 0.0
           else:
               score = 0.0
       except Exception as e:
           print(f"issue fetching traffic risk: {e}")
           score = 0.0
       traffic_risks.append(score)
   traffic_risks.append(traffic_risks[-1]) 
   return traffic_risks




def classify_traffic_condition(normal_duration, traffic_duration):
   if normal_duration == 0:
       return "unknown"
   ratio = traffic_duration / normal_duration
   if ratio >= 2.0:
       return "heavy_traffic"
   elif ratio >= 1.5:
       return "moderate_traffic"
   elif ratio >= 1.2:
       return "light_traffic"
   else:
       return "normal_traffic"


def visualize_risk_aware_path(path, risk_scores, detailed_risks):
   if not path or not risk_scores:
       print("No path or risk scores to visualize.")
       return None




   print("generating folium map..")
   m = folium.Map(location=path[0], zoom_start=9)


   for idx, (point, score, (fire_risk, tornado_risk, traffic_risk)) in enumerate(zip(path, risk_scores, detailed_risks)):
       tooltip_text = (
           f"Total Risk: {score:.2f}<br>"
           f"Fire Risk: {fire_risk:.2f}<br>"
           f"Tornado Risk: {tornado_risk:.2f}<br>"
           f"Traffic Risk: {traffic_risk:.2f}")


       folium.CircleMarker(
           location=point,
           radius=4,
           color=risk_color(score),
           fill=True,
           fill_opacity=0.8,
           tooltip=folium.Tooltip(tooltip_text, sticky=True)
       ).add_to(m)


   folium.PolyLine(
       locations=path,
       color="blue",
       weight=5,
       opacity=0.7,
       popup="Safest Path"
   ).add_to(m)


   for point, score in zip(path, risk_scores):
       if score >= 0.75:
           folium.Circle(
               location=point,
               radius=150,
               color='darkred',
               fill=True,
               fill_opacity=0.3,
               popup="High Risk Zone").add_to(m)
   return m




def risk_color(score: int) -> str:
   if score >= 0.75:
       return 'darkred'
   elif score >= 0.5:
       return 'orange'
   elif score >= 0.25:
       return 'yellow'
   else:
       return 'green'


def compute_bounding_box(path):
   lats, lons = zip(*path)
   return min(lats), min(lons), max(lats), max(lons)


#encode the input address into coords
def geocode_address(address):
   try:
       geocode_result = gmaps.geocode(address)
       if geocode_result:
           location = geocode_result[0]['geometry']['location']
           latitude = location['lat']
           longitude = location['lng']
           print(f" Geocoded '{address}'  ({latitude}, {longitude})")
           return latitude, longitude
       else:
           print(f" No geocoding result for '{address}'")
           return None, None
   except Exception as e:
       print(f" Geocoding failed for '{address}': {e}")
       return None, None






def generate_risk_map(origin_address, destination_address, save_path="risk_path.html"):
   start_lat, start_lon = geocode_address(origin_address)
   goal_lat, goal_lon = geocode_address(destination_address)


   if start_lat is None or goal_lat is None:
       print("Could not get valid coordinates. Exiting function.")
       return None


   start_coords = (start_lat, start_lon)
   goal_coords = (goal_lat, goal_lon)


   print(f"Start: {start_coords}")
   print(f"Goal:  {goal_coords}")
   print("Starting Risk-Aware overlay")


   # Create bounding box with padding
   padding = 0.5
   bbox = (
       min(start_coords[1], goal_coords[1]) - padding,
       min(start_coords[0], goal_coords[0]) - padding,
       max(start_coords[1], goal_coords[1]) + padding,
       max(start_coords[0], goal_coords[0]) + padding
   )


   path = get_google_traffic_data(start_coords, goal_coords)
   if not path:
       print("Cannot get path data")
       return None


   fires = fetch_firms_fires(bbox)
   tornadoes = fetch_tornado_alerts()


   risk_scores = []
   detailed_risks = []


   for idx, (lat, lon) in enumerate(path):
       fire_risk = 0.0
       tornado_risk = 0.0
       traffic_risk = 0.0


       # Check for fire risk
       for fire in fires:
           if abs(fire["latitude"] - lat) < 0.05 and abs(fire["longitude"] - lon) < 0.05:
               fire_risk += 0.4


       # Check for tornado risk
       for poly in tornadoes:
           for pt in poly:
               if abs(pt[1] - lat) < 0.05 and abs(pt[0] - lon) < 0.05:
                   tornado_risk += 0.6


       # Simple placeholder for traffic risk
       if idx < len(path) // 3:
           traffic_risk = 0.2
       elif idx < 2 * len(path) // 3:
           traffic_risk = 0.4
       else:
           traffic_risk = 0.6


       total_risk = min(fire_risk + tornado_risk + traffic_risk, 1.0)
       risk_scores.append(total_risk)
       detailed_risks.append((fire_risk, tornado_risk, traffic_risk))


   m = visualize_risk_aware_path(path, risk_scores, detailed_risks)


   if m:
       m.save(save_path)
       print(f"Map saved as {save_path}")
       return save_path


   return None






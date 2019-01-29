'''
Created on 29 de set de 2015

@author: Alexandre Yukio Yamashita
'''
import pygmaps
import pandas as pd
import numpy as np

path = 'resources/crimes.csv'
# path = 'resources/crimes_testing_samples.csv'
data = pd.read_csv(path, quotechar = '"', skipinitialspace = True)
data = data.as_matrix()
x_y = np.array(data[:, 7:9])

a = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset='utf-8'>
    <title>Heatmaps</title>
    <style>
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
      #map {
        height: 100%;
      }
#floating-panel {
  position: absolute;
  top: 10px;
  left: 25%;
  z-index: 5;
  background-color: #fff;
  padding: 5px;
  border: 1px solid #999;
  text-align: center;
  font-family: 'Roboto','sans-serif';
  line-height: 30px;
  padding-left: 10px;
}

      #floating-panel {
        background-color: #fff;
        border: 1px solid #999;
        left: 25%;
        padding: 5px;
        position: absolute;
        top: 10px;
        z-index: 5;
      }
    </style>
  </head>

  <body>
    <div id="floating-panel">
      <button onclick="toggleHeatmap()">Toggle Heatmap</button>
      <button onclick="changeGradient()">Change gradient</button>
      <button onclick="changeRadius()">Change radius</button>
      <button onclick="changeOpacity()">Change opacity</button>
    </div>
    <div id="map"></div>
    <script>

var map, heatmap;

function initMap() {
  map = new google.maps.Map(document.getElementById('map'), {
    zoom: 13,
    center: {lat: 37.775, lng: -122.434},
    mapTypeId: google.maps.MapTypeId.SATELLITE
  });

  heatmap = new google.maps.visualization.HeatmapLayer({
    data: getPoints(),
    map: map
  });
}

function toggleHeatmap() {
  heatmap.setMap(heatmap.getMap() ? null : map);
}

function changeGradient() {
  var gradient = [
    'rgba(0, 255, 255, 0)',
    'rgba(0, 255, 255, 1)',
    'rgba(0, 191, 255, 1)',
    'rgba(0, 127, 255, 1)',
    'rgba(0, 63, 255, 1)',
    'rgba(0, 0, 255, 1)',
    'rgba(0, 0, 223, 1)',
    'rgba(0, 0, 191, 1)',
    'rgba(0, 0, 159, 1)',
    'rgba(0, 0, 127, 1)',
    'rgba(63, 0, 91, 1)',
    'rgba(127, 0, 63, 1)',
    'rgba(191, 0, 31, 1)',
    'rgba(255, 0, 0, 1)'
  ]
  heatmap.set('gradient', heatmap.get('gradient') ? null : gradient);
}

function changeRadius() {
  heatmap.set('radius', heatmap.get('radius') ? null : 35);
}

function changeOpacity() {
  heatmap.set('opacity', heatmap.get('opacity') ? null : 0.8);
}

// Heatmap data: 500 Points
function getPoints() {
  return [
"""
x_y = x_y[0:500000]
for index in range(len(x_y)):
    x = x_y[index][0]
    y = x_y[index][1]

    if index != len(x_y):
        line = "new google.maps.LatLng(" + str(y) + "," + str(x) + "),"
    else:
        line = "new google.maps.LatLng(" + str(y) + "," + str(x) + "),"

    a += line
a += """
];
}

    </script>
    <script async defer
        src="https://maps.googleapis.com/maps/api/js?libraries=visualization&callback=initMap">
    </script>
  </body>
</html>"""

file = open("heat.html", "w")
file.write(a)
file.close()

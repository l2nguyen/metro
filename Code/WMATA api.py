########### Python 2.7 #############
import httplib
import urllib
import base64
 
headers = {
    # Basic Authorization Sample
    # 'Authorization': 'Basic %s' % base64.encodestring('{username}:{password}'),
}
 
params = urllib.urlencode({
    # Specify your subscription key
    "api_key" : "kfgpmgvfgacx98de9q3xazww"
})

try:
    conn = httplib.HTTPSConnection('api.wmata.com')
    conn.request("GET", "/Incidents.svc/json/Incidents?%s" %(params), "", headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
    

 
headers = {
    # Basic Authorization Sample
    # 'Authorization': 'Basic %s' % base64.encodestring('{username}:{password}'),
}
 
params = urllib.urlencode({
    # Specify your subscription key
    'api_key': 'kfgpmgvfgacx98de9q3xazww',
    # Specify values for optional parameters, as needed
    'Route': 'X2',
})
 
try:
    conn = httplib.HTTPSConnection('api.wmata.com')
    conn.request("GET", "/Incidents.svc/json/BusIncidents?%s" % params, "", headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
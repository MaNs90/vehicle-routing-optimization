from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit
from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sqlalchemy import create_engine
from scipy.spatial.distance import pdist,cdist
from math import radians, cos, sin, asin, sqrt
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
# import googlemaps
import os
import folium
import folium.plugins
import urllib.request as urlrequest
import requests
import json
import time
from folium.features import DivIcon
from folium.plugins import MarkerCluster
import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
from django.conf import settings


def send_mail(send_from='', send_to=[], subject='', message='', files=[],
              server="localhost", port=587, username='', password='',
              use_tls=True):
    """Compose and send email with provided info and attachments.

    Args:
        send_from (str): from name
        send_to (list[str]): to name(s)
        subject (str): message title
        message (str): message body
        files (list[str]): list of file paths to be attached to email
        server (str): mail server host name
        port (int): port number
        username (str): server auth username
        password (str): server auth password
        use_tls (bool): use TLS mode
    """
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message))

    for path in files:
        part = MIMEBase('application', "octet-stream")
        with open(path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename={}'.format(Path(path).name))
        msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    if use_tls:
        smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()

def haversine(p1, p2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    lat1,lon1 = p1
    lat2, lon2 = p2
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def create_distance_matrix_ors(data):
    addresses = data
    API_key = settings.ORS_KEY
    # API only accepts 3500 elements per request, so get rows in multiple requests.
    max_elements = 3500
    num_addresses = len(addresses)
    print('API Matrix request addresses: ', num_addresses)
    distance_matrix = []
    duration_matrix = []

    max_cols = num_addresses
    max_rows, _ = divmod(max_elements, num_addresses)
    q, r = divmod(num_addresses, max_cols)
    a, b = divmod(num_addresses, max_rows)
    for i in range(a):
        # origin_addresses = addresses[i]
        origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
        dest_addresses = addresses
        body = {"locations":addresses,
                "sources":list(range(i * max_rows, (i + 1) * max_rows)),
                # "destinations":dest_addresses,
                "metrics":["distance"]}

        headers = {
            'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
            'Authorization': API_key,
            'Content-Type': 'application/json; charset=utf-8'
        }
        response = requests.post('https://api.openrouteservice.org/v2/matrix/driving-hgv', json=body, headers=headers)
        # print(response.text)
        distance_matrix += json.loads(response.text)['distances']
        # To avoid rate limit exceeded of 40 requests/minute
        time.sleep(2)
    if b > 0:
        origin_addresses = addresses[a * max_rows: a * max_rows + b]
        dest_addresses = addresses
        body = {"locations":addresses,
                "sources":list(range(a * max_rows, a * max_rows + b)),
                # "destinations":dest_addresses,
                "metrics":["distance"]}

        headers = {
            'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
            'Authorization': API_key,
            'Content-Type': 'application/json; charset=utf-8'
        }
        response = requests.post('https://api.openrouteservice.org/v2/matrix/driving-hgv', json=body, headers=headers)
        distance_matrix += json.loads(response.text)['distances']
        # To avoid rate limit exceeded of 40 requests/minute
        time.sleep(2)

    return distance_matrix


def get_product_details():
    rsdb_engine = create_engine(settings.TLDB_CONN, pool_pre_ping=True)
    with rsdb_engine.connect() as conn, conn.begin():
        
        SQL = '''
        ***
        '''
        products_df = pd.read_sql(SQL,conn)

    products_df = products_df[['product_id','product_number','product_name',
                                'category','brand','case_content_description']].copy()
    return products_df

def get_delivery_orders(delivery_date):
    # Replaced hashtag in the connection string password with %23 as the hashtag character has no escape
    rsdb_engine = create_engine(settings.TLDB_CONN, pool_pre_ping=True)
    with rsdb_engine.connect() as conn, conn.begin():
        
        SQL = '''
        ***
        '''
        orders_df = pd.read_sql(SQL,conn)

    orders_df['is_belong'] = orders_df['is_belong'].fillna('0')
    orders_df['order_coordinates'] = orders_df['order_coordinates'].fillna('0')
    orders_df[['order_latitude','order_longitude']] = pd.DataFrame(orders_df['order_coordinates'].str.split(',').tolist(), index= orders_df.index)
    orders_df[['order_latitude','order_longitude']] = orders_df[['order_latitude','order_longitude']].astype(float)
    orders_df.loc[orders_df['latitude'].isna(),'is_belong'] = 0
    orders_df.loc[orders_df['longitude'].isna(),'is_belong'] = 0
    orders_df.loc[orders_df['order_latitude']==0,['is_belong','location_matched']] = [0,0]
    orders_df.loc[orders_df['order_longitude']==0,['is_belong','location_matched']] = [0,0]
    orders_df['second_leg'] = np.where(orders_df['delivery_period']==2,1,0)

    return orders_df

def get_subregion_coordinates():
    rsdb_engine = create_engine(settings.TLDB_CONN, pool_pre_ping=True)
    with rsdb_engine.connect() as conn, conn.begin():
        
        SQL = '''
        ***
        '''
        sub_region_coordinates = pd.read_sql(SQL,conn)

    return sub_region_coordinates

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


def draw_map(folium_df,vehicle_routes):
    colors = [
        'red',
        'blue',
        'gray',
        'darkred',
        'lightred',
        'orange',
        # 'beige',
        'green',
        'darkgreen',
        'lightgreen',
        'darkblue',
        'lightblue',
        'purple',
        'darkpurple',
        'pink',
        'cadetblue',
        'lightgray',
        'black'
    ]

    m = folium.Map(location=[30.02599061, 31.18720606], zoom_start=10, tiles='cartodbpositron')
    count = 0
    mc = MarkerCluster()
    folium.Marker(WAREHOUSE_COORDINATES, icon=folium.Icon(color='green')).add_to(m)
    for i, bin_id in enumerate(vehicle_routes.keys()):
        i += 1
        if vehicle_routes[bin_id] == None:
            continue
        color = colors[count % len(colors)]
        client_ids_order = vehicle_routes[bin_id]
        # print(client_ids_order)
        custom_dict = dict(zip(vehicle_routes[bin_id],range(len(vehicle_routes[bin_id]))))
        route = folium_df[folium_df['client_id'].isin(client_ids_order)]
        route = route.drop_duplicates('client_id',keep='last')
        route = route.sort_values('client_id',key=lambda x: x.map(custom_dict))
        bin_group = folium.FeatureGroup(name=i).add_to(m)
        if route['order_latitude'].isna().sum()>0:
            continue
        route_points = route[['order_latitude', 'order_longitude']].values.tolist()
        route_points = [WAREHOUSE_COORDINATES] + route_points
        return_points = [i for i, v in enumerate(vehicle_routes[bin_id]) if v == 'warehouse' and i not in [0,len(vehicle_routes[bin_id])-1]]
        for r in return_points:
            route_points.insert(r,WAREHOUSE_COORDINATES)
        dispatch_loc = route[['order_latitude', 'order_longitude']].iloc[0].values.tolist()
        # loc_at_accept = route[route.arriving == 0].iloc[0][['latitude', 'longitude']].values
        loc_at_end = route.iloc[-1][['order_latitude', 'order_longitude']].values.tolist()
        # for p in route_points:
        #     folium.Marker(p, icon=folium.Icon(color=color)).add_to(m)
        bin_group.add_child(folium.PolyLine(route_points, 
                    weight=2, 
                    color=color,
                   popup=i))
        count2 = 0
        for loc in route[['order_latitude', 'order_longitude']].values.tolist():
            bin_group.add_child(folium.CircleMarker(loc, color =color, fill=True, radius=5, popup='{}\n{}'.format(route['sub_region_name'].values[count2],i)))
            count2+=1
        # folium.CircleMarker(dispatch_loc, color='red', fill=True, radius=3, popup='Dispatch Location').add_to(m)
        # # folium.CircleMarker(loc_at_accept, color='green', fill=True, radius=3, popup='Start Trip').add_to(m)
        # folium.CircleMarker(loc_at_end, color='darkblue', fill=True, radius=3, popup='End Trip').add_to(m)
        # folium.CircleMarker(trips_all[trips_all['trip_id'] == trip_id][['request_latitude','request_longitude']].values, color='black', fill=True, radius=5, popup='Request Trip {}'.format(trip_id)).add_to(m)
        count += 1
    folium.LayerControl().add_to(m)
    sw = [30.036962, 31.197045]
    ne = [30.191207, 31.294183]
    m.fit_bounds([sw, ne])  
    m.add_child(folium.plugins.MeasureControl())

    return m


def build_vehicle_route(manager, routing, plan, data, veh_number, nasrcity_test):
    """
    Build a route for a vehicle by starting at the strat node and
    continuing to the end node.
    Args: routing (ortools.constraint_solver.pywrapcp.RoutingModel): routing.
    plan (ortools.constraint_solver.pywrapcp.Assignment): the assignment.
    customers (Customers): the customers instance.  veh_number (int): index of
    the vehicle
    Returns:
        (List) route: indexes of the customers for vehicle veh_number
    """
    veh_used = routing.IsVehicleUsed(plan, veh_number)
    print('Vehicle {0} is used {1}'.format(veh_number, veh_used))
    if veh_used:
        route = []
        subregions = []
        secondrun = False
        node = routing.Start(veh_number)  # Get the starting node index
        # route.append(data['clients'][manager.IndexToNode(node)])
        while not routing.IsEnd(node):
            route.append(data['clients'][manager.IndexToNode(node)])
            subregions.append(data['subregions'][manager.IndexToNode(node)])
            node = plan.Value(routing.NextVar(node))
            if manager.IndexToNode(node) in range(1,(VAN_COUNT+TRICYCLE_COUNT+1)):
                secondrun = True
                nasrcity_test.loc[nasrcity_test['client_id'].isin(route),'bin_id_final'] = veh_number
                subregions_firstrun = subregions.copy()
                route_firstrun = route.copy()
                # route2 = []
                # subregions2 = []

        route.append(data['clients'][manager.IndexToNode(node)])
        subregions.append(data['subregions'][manager.IndexToNode(node)])
        if secondrun:
            
            route_secondrun = list(set(route) - set(route_firstrun))
            print(set(route))
            print(set(route_firstrun))
            print(route_secondrun)
            nasrcity_test.loc[nasrcity_test['client_id'].isin(route_secondrun),'bin_id_final'] = veh_number + data['num_vehicles'] # account for second run
            # return route_firstrun,route,subregions_firstrun,subregions,nasrcity_test
        else:
            nasrcity_test.loc[nasrcity_test['client_id'].isin(route),'bin_id_final'] = veh_number
        return route,subregions,nasrcity_test
    else:
        return None


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    total_value = 0
    total_orders = 0
    total_second_legs = 0
    route_distance_list = []
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    value_dimension = routing.GetDimensionOrDie('Value')
    counter_dimension = routing.GetDimensionOrDie('Counter')
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        # route_load = 0
        route_value = 0
        orders_count = 0
        second_leg_count = 0
        second_legs = []
        subregions = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            load_var = capacity_dimension.CumulVar(index)
            order_var = counter_dimension.CumulVar(index)
            # value_var = value_dimension.CumulVar(index)
            # route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, solution.Value(load_var))
            route_value += data['values'][node_index]
            # plan_output += ' {0} Value({1}) -> '.format(node_index, route_value)
            # orders_count += data['orders'][node_index]
            # print(node_index)
            # print(len(data['second_leg']))
            second_leg_count += data['second_leg'][node_index]
            second_legs.append(data['clients'][node_index])
            subregions.append(data['subregions'][node_index])
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        load_var = capacity_dimension.CumulVar(index)
        order_var = counter_dimension.CumulVar(index)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 solution.Value(load_var))
        plan_output += ' {0} Value({1})\n'.format(manager.IndexToNode(index),
                                                 route_value)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(solution.Value(load_var))
        plan_output += 'Value of the route: {}\n'.format(route_value)
        plan_output += 'Number of orders of the route: {}\n'.format(solution.Value(order_var))
        plan_output += 'Number of second leg orders of the route: {}\n'.format(second_leg_count)
        print(plan_output)
        # print(second_legs)
        # print(set(subregions))
        total_distance += route_distance
        total_load += solution.Value(load_var)
        total_value += route_value
        total_orders += solution.Value(order_var)
        total_second_legs += second_leg_count
        route_distance_list.append(route_distance)
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))
    print('Total Value of all routes: {}'.format(total_value))
    print('Total Orders of all routes: {}'.format(total_orders))
    print('Total Second Leg Orders of all routes: {}'.format(total_second_legs))
    return route_distance_list

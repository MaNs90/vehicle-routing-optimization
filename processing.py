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


def process_region(orders_df,sub_region_coordinates):
    orders_region_df = orders_df.sort_values(['region_id','sub_region_id','order_id'])
    orders_region_df[['latitude','longitude']] = orders_region_df[['latitude','longitude']].astype(float)
    orders_region_df = orders_region_df.merge(sub_region_coordinates.reset_index(), on='sub_region_id',how='left')
    orders_region_df['latitude'] = np.where(orders_region_df['subregion_latitude'].isna()==False,
                                                          orders_region_df['subregion_latitude'],
                                                          orders_region_df['latitude'])
    orders_region_df['longitude'] = np.where(orders_region_df['subregion_longitude'].isna()==False,
                                                          orders_region_df['subregion_longitude'],
                                                          orders_region_df['longitude'])
    
    orders_region_df['order_latitude'] = np.where((orders_region_df['location_matched']==0) &
                                                  (orders_region_df['is_belong']==0) & 
                                                  (orders_region_df['subregion_latitude'].isna()==False),
                                                                      orders_region_df['subregion_latitude'],
                                                                      np.where((orders_region_df['location_matched']==0) &
                                                                                (orders_region_df['is_belong']==0) & 
                                                                                (orders_region_df['subregion_latitude'].isna()),orders_region_df['latitude'],
                                                                                orders_region_df['order_latitude']))
    orders_region_df['order_longitude'] = np.where((orders_region_df['location_matched']==0) &
                                                  (orders_region_df['is_belong']==0) & 
                                                  (orders_region_df['subregion_longitude'].isna()==False),
                                                                      orders_region_df['subregion_longitude'],
                                                                      np.where((orders_region_df['location_matched']==0) &
                                                                                (orders_region_df['is_belong']==0) & 
                                                                                (orders_region_df['subregion_longitude'].isna()),orders_region_df['longitude'],
                                                                                orders_region_df['order_longitude']))

    orders_region_df['order_latitude'] = np.where(orders_region_df['order_latitude'].isna(),
                                                    orders_region_df['order_coordinates'].str.split(',').str[0].astype(float),orders_region_df['order_latitude'])
    orders_region_df['order_longitude'] = np.where(orders_region_df['order_longitude'].isna(),
                                                    orders_region_df['order_coordinates'].str.split(',').str[1].astype(float),orders_region_df['order_longitude'])
    
    orders_region_df['latitude'] = np.where(orders_region_df['latitude'].isna(),
                                                    orders_region_df['order_latitude'],orders_region_df['latitude'])
    orders_region_df['longitude'] = np.where(orders_region_df['longitude'].isna(),
                                                    orders_region_df['order_longitude'],orders_region_df['longitude'])

    orders_region_df['latitude'] = orders_region_df['latitude'].fillna(WAREHOUSE_COORDINATES[0])
    orders_region_df['longitude'] = orders_region_df['longitude'].fillna(WAREHOUSE_COORDINATES[1])
    orders_region_df['order_latitude'] = orders_region_df['order_latitude'].fillna(WAREHOUSE_COORDINATES[0])
    orders_region_df['order_longitude'] = orders_region_df['order_longitude'].fillna(WAREHOUSE_COORDINATES[1])

    orders_region_df['latitude'] = np.where(orders_region_df['latitude']==0,WAREHOUSE_COORDINATES[0],orders_region_df['latitude'])
    orders_region_df['longitude'] = np.where(orders_region_df['longitude']==0,WAREHOUSE_COORDINATES[1],orders_region_df['longitude'])
    orders_region_df['order_latitude'] = np.where(orders_region_df['order_latitude']==0,WAREHOUSE_COORDINATES[0],orders_region_df['order_latitude'])
    orders_region_df['order_longitude'] = np.where(orders_region_df['order_longitude']==0,WAREHOUSE_COORDINATES[1],orders_region_df['order_longitude'])
    
    distance_df = orders_region_df.sort_values(['region_id','sub_region_id','order_id'])[['sub_region_id','latitude','longitude']].groupby(['sub_region_id'],sort=False)[['latitude','longitude']].max().reset_index(drop=True)
    dist_matrix = cdist(distance_df, 
          distance_df, 
          haversine)
    dist_matrix = (dist_matrix*SCALING_FACTOR).round().astype(int)
    # To account for wrong coordinates!
    # dist_matrix[dist_matrix>100*SCALING_FACTOR]=5*SCALING_FACTOR
    dist_matrix = dist_matrix.tolist()
    # Uncomment it if you will work on item level not order level
    orders_region_df.loc[orders_region_df['product_weight']==0,'product_weight'] = 0.1
    return orders_region_df, dist_matrix

def create_data_model(orders_region_df,dist_matrix):
    """Create the data for the example."""
    data = {}
    # orders_region_df = orders_region_df[orders_region_df['client_id']!='764D79DC-4D26-419C-A460-46BE429D5702'].copy()
    orders_region_df['product_weight'] = orders_region_df['product_weight'] * SCALING_FACTOR
    orders_region_df['exceptional_subregions'] = np.where(orders_region_df['sub_region_id'].isin(EXCEPTIONAL_SUBREGIONS),1,0)
    client_weights = orders_region_df.groupby('client_id').agg({'product_weight':'sum','exceptional_subregions':'max'})
    client_ids = 0
    mask1 = (client_weights['product_weight'] > VAN_WEIGHT*SCALING_FACTOR)
    mask2 = ((client_weights['product_weight'] > TRICYCLE_WEIGHT*SCALING_FACTOR) & 
             (client_weights['product_weight'] < VAN_WEIGHT*SCALING_FACTOR) &  
             (client_weights['exceptional_subregions'] == 1))
    if (mask1.sum()>0) | (mask2.sum()>0):
        client_ids = client_weights[mask1 | mask2].index
        dedicated_orders = orders_region_df[orders_region_df['client_id'].isin(client_ids)].copy()
        orders_region_df = orders_region_df[~orders_region_df['client_id'].isin(client_ids)].copy()
        print('Exception Clients: ',client_ids)
    else:
        dedicated_orders = pd.DataFrame()
        
        print(orders_region_df.shape)
    # weights = [48, 30, 19, 36, 36, 27, 42, 42, 36, 24, 30]
    # volumes = [20, 40, 60, 12, 5, 6, 10, 21, 33, 11, 72]
    # values  = [20, 40, 60, 12, 5, 6, 10, 21, 33, 11, 72]
    
    values = orders_region_df.sort_values(['region_id','sub_region_id','order_id'])[['product_value',
                                                                                 'product_weight']]['product_value'].values.round().astype(int).tolist()
    weights = orders_region_df.sort_values(['region_id','sub_region_id','order_id'])[['product_value',
                                                                                 'product_weight']]['product_weight'].values.round().astype(int).tolist()
    orders = orders_region_df.sort_values(['region_id','sub_region_id','order_id']).groupby('order_id',sort=False).ngroup().values.tolist()
    
    second_leg = orders_region_df.sort_values(['region_id','sub_region_id','order_id'])['second_leg'].values.tolist()
    
    clients = orders_region_df.sort_values(['region_id','sub_region_id','order_id']).groupby('client_id',sort=False).ngroup().values.tolist()
    
    regions = orders_region_df.sort_values(['region_id','sub_region_id','order_id']).groupby('region_id',sort=False).ngroup().values.tolist()
    
    subregions = orders_region_df.sort_values(['region_id','sub_region_id','order_id']).groupby('sub_region_id',sort=False).ngroup().values.tolist()

    conditional_subregions = orders_region_df.sort_values(['region_id','sub_region_id','order_id'])['exceptional_subregions'].values.tolist()

    # The maxbins saved my life because its a main factor of runtime, the initial solution was that each item is in it's own bin
    # which is ridiculous
    max_bins = DABABA_COUNT + (VAN_COUNT*2) + (TRICYCLE_COUNT*2)
    data['max_distance'] = np.max(dist_matrix)
    data['weights'] = weights 
    # data['volumes'] = volumes
    data['values'] = values
    data['items'] = list(range(len(weights)))
    data['second_leg'] = second_leg
    data['orders'] = orders
    data['clients'] = clients
    data['regions'] = regions
    data['subregions'] = subregions
    data['conditional_subregions'] = conditional_subregions
    data['bins'] = data['items'][:max_bins]
    
    cap_0 = [DABABA_WEIGHT * SCALING_FACTOR] * DABABA_COUNT
    cap_1 = [VAN_WEIGHT * SCALING_FACTOR] * (VAN_COUNT*2)
    cap_2 = [TRICYCLE_WEIGHT * SCALING_FACTOR] * (TRICYCLE_COUNT*2)
    data['bin_capacity'] = cap_0 + cap_1 + cap_2# 800 for tricycle
    cost_0 = [DABABA_PRICE * SCALING_FACTOR] * DABABA_COUNT
    cost_1 = [VAN_PRICE] * (VAN_COUNT*2)  # 360/2
    cost_2 = [TRICYCLE_PRICE] * (TRICYCLE_COUNT*2)  # 280/3
    data['bin_costs'] = cost_0 + cost_1 + cost_2 # 800 for tricycle
    ord_0 = [DABABA_ORDERS * SCALING_FACTOR] * DABABA_COUNT
    ord_1 = [VAN_ORDERS] * (VAN_COUNT*2)
    ord_2 = [TRICYCLE_ORDERS] * (TRICYCLE_COUNT*2)
    data['bin_orders'] = ord_0 + ord_1 + ord_2
    val_0 = [DABABA_VALUE] * DABABA_COUNT
    val_1 = [VAN_VALUE] * (VAN_COUNT*2)
    val_2 = [TRICYCLE_VALUE] * (TRICYCLE_COUNT*2)
    data['values_constraint'] = val_0 + val_1 + val_2

    return data,orders_region_df, dedicated_orders



def create_data_model_routing(orders_df,orders_region_df_opt,sub_region_coordinates):
    """Stores the data for the problem."""
    orders_region_df = orders_df.sort_values(['region_id','sub_region_id','order_id'])
    orders_region_df[['latitude','longitude']] = orders_region_df[['latitude','longitude']].astype(float)
    orders_region_df = orders_region_df.merge(sub_region_coordinates.reset_index(), on='sub_region_id',how='left')
    orders_region_df['latitude'] = np.where(orders_region_df['subregion_latitude'].isna()==False,
                                                          orders_region_df['subregion_latitude'],
                                                          orders_region_df['latitude'])
    orders_region_df['longitude'] = np.where(orders_region_df['subregion_longitude'].isna()==False,
                                                          orders_region_df['subregion_longitude'],
                                                          orders_region_df['longitude'])
    
    orders_region_df['order_latitude'] = np.where((orders_region_df['location_matched']==0) &
                                                  (orders_region_df['is_belong']==0) & 
                                                  (orders_region_df['subregion_latitude'].isna()==False),
                                                                      orders_region_df['subregion_latitude'],
                                                                      np.where((orders_region_df['location_matched']==0) &
                                                                                (orders_region_df['is_belong']==0) & 
                                                                                (orders_region_df['subregion_latitude'].isna()),orders_region_df['latitude'],
                                                                                orders_region_df['order_latitude']))
    orders_region_df['order_longitude'] = np.where((orders_region_df['location_matched']==0) &
                                                  (orders_region_df['is_belong']==0) & 
                                                  (orders_region_df['subregion_longitude'].isna()==False),
                                                                      orders_region_df['subregion_longitude'],
                                                                      np.where((orders_region_df['location_matched']==0) &
                                                                                (orders_region_df['is_belong']==0) & 
                                                                                (orders_region_df['subregion_longitude'].isna()),orders_region_df['longitude'],
                                                                                orders_region_df['order_longitude']))

    orders_region_df['order_latitude'] = np.where(orders_region_df['order_latitude'].isna(),
                                                    orders_region_df['order_coordinates'].str.split(',').str[0].astype(float),orders_region_df['order_latitude'])
    orders_region_df['order_longitude'] = np.where(orders_region_df['order_longitude'].isna(),
                                                    orders_region_df['order_coordinates'].str.split(',').str[1].astype(float),orders_region_df['order_longitude'])
    
    orders_region_df['latitude'] = np.where(orders_region_df['latitude'].isna(),
                                                    orders_region_df['order_latitude'],orders_region_df['latitude'])
    orders_region_df['longitude'] = np.where(orders_region_df['longitude'].isna(),
                                                    orders_region_df['order_longitude'],orders_region_df['longitude'])

    orders_region_df['latitude'] = orders_region_df['latitude'].fillna(WAREHOUSE_COORDINATES[0])
    orders_region_df['longitude'] = orders_region_df['longitude'].fillna(WAREHOUSE_COORDINATES[1])
    orders_region_df['order_latitude'] = orders_region_df['order_latitude'].fillna(WAREHOUSE_COORDINATES[0])
    orders_region_df['order_longitude'] = orders_region_df['order_longitude'].fillna(WAREHOUSE_COORDINATES[1])

    orders_region_df['latitude'] = np.where(orders_region_df['latitude']==0,WAREHOUSE_COORDINATES[0],orders_region_df['latitude'])
    orders_region_df['longitude'] = np.where(orders_region_df['longitude']==0,WAREHOUSE_COORDINATES[1],orders_region_df['longitude'])
    orders_region_df['order_latitude'] = np.where(orders_region_df['order_latitude']==0,WAREHOUSE_COORDINATES[0],orders_region_df['order_latitude'])
    orders_region_df['order_longitude'] = np.where(orders_region_df['order_longitude']==0,WAREHOUSE_COORDINATES[1],orders_region_df['order_longitude'])
    
    orders_region_df['product_weight'] = orders_region_df['product_weight'] * SCALING_FACTOR
    orders_region_df['exceptional_subregions'] = np.where(orders_region_df['sub_region_id'].isin(EXCEPTIONAL_SUBREGIONS),1,0)   
    
    client_weights = orders_region_df.groupby('client_id').agg({'product_weight':'sum','exceptional_subregions':'max'})
    client_ids = 0
    mask1 = (client_weights['product_weight'] > VAN_WEIGHT*SCALING_FACTOR)
    mask2 = ((client_weights['product_weight'] > TRICYCLE_WEIGHT*SCALING_FACTOR) & 
             (client_weights['product_weight'] < VAN_WEIGHT*SCALING_FACTOR) &  
             (client_weights['exceptional_subregions'] == 1))
    if (mask1.sum()>0) | (mask2.sum()>0):
        client_ids = client_weights[mask1 | mask2].index
        dedicated_orders = orders_region_df[orders_region_df['client_id'].isin(client_ids)].copy()
        orders_region_df = orders_region_df[~orders_region_df['client_id'].isin(client_ids)].copy()
        print('Exception Clients: ',client_ids)
    else:
        dedicated_orders = []
    
    demand = orders_region_df.sort_values(['region_id','sub_region_id','client_id']).groupby(['client_id'],sort=False).agg({'product_weight':'sum'})['product_weight'].astype(int).values.tolist()
    demand = [0] + [-VAN_WEIGHT*SCALING_FACTOR]*VAN_COUNT + [-TRICYCLE_WEIGHT*SCALING_FACTOR]*TRICYCLE_COUNT + demand
    values = orders_region_df.sort_values(['region_id','sub_region_id','client_id']).groupby(['client_id'],sort=False).agg({'product_value':'sum'})['product_value'].astype(int).values.tolist()
    values = [0] + [0]*VAN_COUNT + [0]*TRICYCLE_COUNT + values
    orders = orders_region_df.sort_values(['region_id','sub_region_id','client_id']).groupby(['client_id'],sort=False)['order_id'].nunique().values.tolist()
    orders = [1 if o > 1 else o for o in orders]
    orders = [0] + [-VAN_ORDERS]*VAN_COUNT + [-TRICYCLE_ORDERS]*TRICYCLE_COUNT + orders
    second_leg = orders_region_df.sort_values(['region_id','sub_region_id','client_id']).groupby(['client_id'],sort=False)['second_leg'].max().values.tolist()
    second_leg = [0] + [0]*VAN_COUNT + [0]*TRICYCLE_COUNT + second_leg
    # second_leg_series = pd.Series(second_leg)
    # second_leg_series.loc[7:] = 0
    # second_leg = second_leg_series.values.tolist()
    clients = orders_region_df.sort_values(['region_id','sub_region_id','client_id']).groupby(['client_id'],sort=False).size().index.tolist()
    clients = ['warehouse'] + ['warehouse']*VAN_COUNT + ['warehouse']*TRICYCLE_COUNT + clients
    subregions = orders_region_df.sort_values(['region_id','sub_region_id','client_id']).groupby(['client_id'],sort=False)['sub_region_name'].first().values.tolist()
    subregions = ['warehouse'] + ['warehouse']*VAN_COUNT + ['warehouse']*TRICYCLE_COUNT + subregions
    regions = orders_region_df.sort_values(['region_id','sub_region_id','client_id']).groupby(['client_id'],sort=False)['region_name'].first().values.tolist()
    regions = ['warehouse'] + ['warehouse']*VAN_COUNT + ['warehouse']*TRICYCLE_COUNT + regions
    conditional_subregions = orders_region_df.sort_values(['region_id','sub_region_id','client_id']).groupby(['client_id'],sort=False)['exceptional_subregions'].max().values.tolist()
    conditional_subregions = [0] + [0]*VAN_COUNT + [0]*TRICYCLE_COUNT + conditional_subregions
    distance_df = orders_region_df.sort_values(['region_id','sub_region_id','client_id'])[['client_id','order_latitude','order_longitude']].groupby(['client_id'],sort=False)[['order_latitude','order_longitude']].max().reset_index(drop=True)
    distance_df.loc[-1] = WAREHOUSE_COORDINATES  # shobra hub
    for i in range(2,VAN_COUNT+TRICYCLE_COUNT+2):
        distance_df.loc[-i] = WAREHOUSE_COORDINATES
    distance_df.index = distance_df.index + (VAN_COUNT+TRICYCLE_COUNT+1)  # shifting index
    distance_df = distance_df.sort_index() 
    # openroute service requires lng1 lat1 instead of lat1 lng1
    print('Getting distance matrix from OSM...')
    dist_matrix = create_distance_matrix_ors(distance_df[['order_longitude','order_latitude']].values.tolist())
    dist_matrix = np.array(dist_matrix).astype(int).tolist()
    # Uncomment it if you will work on item level not order level
    orders_region_df.loc[orders_region_df['product_weight']==0,'product_weight'] = 0.1
    print('Number of orders: ', orders_region_df['order_id'].nunique())
    
    cpsat_solution = orders_region_df_opt.groupby(['bin_id'])['client_id'].unique()
    initial_solution = []
    for index, row in cpsat_solution.iteritems():
        initial_solution.append([x + 1 
                                 for x in orders_region_df.sort_values(['region_id',
                                                                        'sub_region_id',
                                                                        'client_id'])
                                 .groupby(['client_id'],sort=False)
                                 .size()
                                 .reset_index()
                                 .index[orders_region_df.sort_values(['region_id',
                                                                      'sub_region_id',
                                                                      'client_id'])
                                        .groupby(['client_id'],sort=False)
                                        .size()
                                        .reset_index()['client_id'].isin(row)]
                                 .values
                                 .tolist()]
                               )
    data = {}
    
    data['initial_solution'] = initial_solution
    # vehicle_0 = (orders_region_df_opt.groupby(['bin_id'])['product_weight'].sum()>VAN_WEIGHT).sum()
    # vehicle_1 = ((orders_region_df_opt.groupby(['bin_id'])['product_weight'].sum()>TRICYCLE_WEIGHT) & 
    #              (orders_region_df_opt.groupby(['bin_id'])['product_weight'].sum()<=VAN_WEIGHT)).sum()
    # vehicle_2 = (orders_region_df_opt.groupby(['bin_id'])['product_weight'].sum()<=TRICYCLE_WEIGHT).sum()
    # vehicle_0 = orders_region_df_opt[orders_region_df_opt['bin_id'] == 0]['bin_id'].nunique()
    # vehicle_1 = orders_region_df_opt[(orders_region_df_opt['bin_id'] > 0) & (orders_region_df_opt['bin_id'] < 5)]['bin_id'].nunique()
    # vehicle_2 = orders_region_df_opt[orders_region_df_opt['bin_id'] >= 5]['bin_id'].nunique()
    # data['vehicle_0'] = vehicle_0
    # data['vehicle_1'] = vehicle_1
    # data['vehicle_2'] = vehicle_2
    data['distance_matrix'] = dist_matrix
    # This part is to nullify the return of the vehicles to the warehouse (Need to ask about that!!)
    dist_matrix_mod = np.array(data['distance_matrix'])
    dist_matrix_mod[:,0] = 0
    data['distance_matrix'] = dist_matrix_mod.tolist()
    data['demands'] = demand
    data['values'] = values
    data['orders'] = orders
    data['second_leg'] = second_leg
    data['clients'] = clients
    data['subregions'] = subregions
    data['regions'] = regions
    data['conditional_subregions'] = conditional_subregions
    bin_vehicles = orders_region_df_opt.groupby(['bin_id'])[['product_weight','product_value']].sum()

    # bin_vehicles['vehicle_capacities'] = np.where(bin_vehicles['product_weight']>VAN_WEIGHT,DABABA_WEIGHT* SCALING_FACTOR,np.where(bin_vehicles['product_weight']>TRICYCLE_WEIGHT,VAN_WEIGHT * SCALING_FACTOR,TRICYCLE_WEIGHT* SCALING_FACTOR))
    # bin_vehicles['vehicle_values'] = np.where(bin_vehicles['product_weight']>VAN_WEIGHT,DABABA_VALUE,np.where(bin_vehicles['product_weight']>TRICYCLE_WEIGHT,VAN_VALUE,TRICYCLE_VALUE))
    # bin_vehicles['vehicle_orders'] = np.where(bin_vehicles['product_weight']>VAN_WEIGHT,DABABA_ORDERS,np.where(bin_vehicles['product_weight']>TRICYCLE_WEIGHT,VAN_ORDERS,TRICYCLE_ORDERS))
    # bin_vehicles['vehicle_capacities'] = np.where(bin_vehicles.reset_index()['bin_id']==0,DABABA_WEIGHT* SCALING_FACTOR,np.where((bin_vehicles.reset_index()['bin_id'] > 0) & (bin_vehicles.reset_index()['bin_id'] < 5),VAN_WEIGHT * SCALING_FACTOR,TRICYCLE_WEIGHT* SCALING_FACTOR))
    # bin_vehicles['vehicle_values'] = np.where(bin_vehicles.reset_index()['bin_id']==0,DABABA_VALUE,np.where((bin_vehicles.reset_index()['bin_id'] > 0) & (bin_vehicles.reset_index()['bin_id'] < 5),VAN_VALUE,TRICYCLE_VALUE))
    # bin_vehicles['vehicle_orders'] = np.where(bin_vehicles.reset_index()['bin_id']==0,DABABA_ORDERS,np.where((bin_vehicles.reset_index()['bin_id'] > 0) & (bin_vehicles.reset_index()['bin_id'] < 5),VAN_ORDERS,TRICYCLE_ORDERS))
    
    # data['vehicle_capacities'] = bin_vehicles['vehicle_capacities'].values.tolist()#cap_1 + cap_2 + cap_3# 800 for tricycle
    # data['vehicle_values'] = bin_vehicles['vehicle_values'].values.tolist() #val_1 + val_2 + val_3
    # data['vehicle_orders'] = bin_vehicles['vehicle_orders'].values.tolist() #ord_1 + ord_2 + ord_3
    # data['num_vehicles'] = orders_region_df_opt['bin_id'].nunique()

    data['vehicle_0'] = DABABA_COUNT
    data['vehicle_1'] = VAN_COUNT
    data['vehicle_2'] = TRICYCLE_COUNT
    data['num_vehicles'] = data['vehicle_0'] + data['vehicle_1'] + data['vehicle_2']
    cap_0 = [DABABA_WEIGHT * SCALING_FACTOR] * DABABA_COUNT
    cap_1 = [VAN_WEIGHT * SCALING_FACTOR] * VAN_COUNT
    cap_2 = [TRICYCLE_WEIGHT * SCALING_FACTOR] * TRICYCLE_COUNT
    data['vehicle_capacities'] = cap_0 + cap_1 + cap_2# 800 for tricycle
    ord_0 = [DABABA_ORDERS * SCALING_FACTOR] * DABABA_COUNT
    ord_1 = [VAN_ORDERS] * VAN_COUNT
    ord_2 = [TRICYCLE_ORDERS] * TRICYCLE_COUNT
    data['vehicle_orders'] = ord_0 + ord_1 + ord_2
    val_0 = [DABABA_VALUE] * DABABA_COUNT
    val_1 = [VAN_VALUE] * VAN_COUNT
    val_2 = [TRICYCLE_VALUE] * TRICYCLE_COUNT
    data['vehicle_values'] = val_0 + val_1 + val_2
    data['depot'] = 0
    
    return data



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

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


def design_model(orders,dist_matrix,data,fallback=False):
    
    # Creates the model.
    model = cp_model.CpModel()
    
    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data['items']:
        for j in data['bins']:
            # x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))
            x[(i, j)] = model.NewBoolVar('x_%i_%i' % (i, j))
    
    #z[k, j] = 1 if an item from order k is packed in bin j.
    # z = {}
    # for k in set(data['orders']):
    #     for j in data['bins']:
    #         z[(k, j)] = model.NewBoolVar('z_%i_%i' % (k, j))
    #         # z[(j, k)] = solver.IntVar(0, 1, 'z_%i_%i' % (j, k))
    
    # # r[m, j] = 1 if an item from region m is packed in bin j.
    # r = {}
    # for m in set(data['regions']):
    #     for j in data['bins']:
    #         r[(m, j)] = model.NewBoolVar('r_%i_%i' % (m, j))
            
    # s[m, j] = 1 if an item from subregion m is packed in bin j.
    s = {}
    for m in set(data['subregions']):
        for j in data['bins']:
            s[(m, j)] = model.NewBoolVar('s_%i_%i' % (m, j))
            # z[(j, m)] = solver.IntVar(0, 1, 'z_%i_%i' % (j, m))
            
    
    # c[h, j] = 1 if client h is packed in bin j.
    c = {}
    for h in set(data['clients']):
        for j in data['bins']:
            c[(h, j)] = model.NewBoolVar('c_%i_%i' % (h, j))
            # z[(j, k)] = solver.IntVar(0, 1, 'z_%i_%i' % (j, k))
    
    # d[j, m, n] = 1 if bin j has items from subregions m and n
    d = {}
    for j in data['bins']:
        for ind,m in enumerate(set(data['subregions'])):
            for n in list(set(data['subregions']))[ind:]:
                d[(j, m, n)] = model.NewBoolVar('d_%i_%i_%i' % (j, m, n))
                
    e = {}
    for ind,m in enumerate(set(data['subregions'])):
        for n in list(set(data['subregions']))[ind:]:
                e[(m, n)] = model.NewBoolVar('e_%i_%i' % (m, n))
    # d = {}
    # for j in data['bins']:
    #     for ind,m in enumerate(data['items']):
    #         for n in list(data['items'])[ind:]:
    #             d[(j, m, n)] = model.NewBoolVar('d_%i_%i_%i' % (j, m, n))
    # d = {}
    # for ind,m in enumerate(set(data['subregions'])):
    #     for n in list(set(data['subregions']))[ind:]:
    #         d[(m, n)] = model.NewIntVar(0,50,'d_%i_%i' % (m, n))

    # y[j] = 1 if bin j is used.
    y = {}
    for j in data['bins']:
        # y[j] = solver.IntVar(0, 1, 'y[%i]' % j)
        y[j] = model.NewBoolVar('y[%i]' % j)
    
    # o[j] = 1 if bin j has a second leg item
    o = {}
    for j in data['bins']:
        # y[j] = solver.IntVar(0, 1, 'y[%i]' % j)
        # o[j] = model.NewIntVar(0,2,'o[%i]' % j)
        o[j] = model.NewBoolVar('o[%i]' % j)
    
    
    # Constraints
    
    # Each item must be in exactly one bin.
    for i in data['items']:
        model.Add(cp_model.LinearExpr.Sum([x[i, j] for j in data['bins']]) == 1)
        # model.AddExactlyOne([x[i, j] for j in data['bins']])

    # The amount packed in each bin cannot exceed its capacity.
    for j in data['bins']:
        model.Add(
            cp_model.LinearExpr.Sum([x[(i, j)] * data['weights'][i] for i in data['items']]) <= y[j] *
            data['bin_capacity'][j])

    # # The amount packed in each bin cannot exceed its volume.
    # for j in data['bins']:
    #     solver.Add(
    #         sum(x[(i, j)] * data['volumes'][i] for i in data['items']) <= y[j] *
    #         data['volume_capacity'])

    #The amount packed in each bin is not less than the value.
    if fallback == False:
        for j in data['bins']:
            model.Add(
                sum(x[(i, j)] * data['values'][i] for i in data['items']) >= y[j] *
                data['values_constraint'][j])

    # # add constraints MaxEquality for y[j, k] as maximum of x[i, j] for all i where data['delivery_county_id'][i] == k to make sure y has the right values
    # for k in set(data['orders']):
    #     for j in data['bins']:
    #         # model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items'] if data['orders'][i] == k ]) >= 1).OnlyEnforceIf(z[k,j])
    #         # model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items'] if data['orders'][i] == k ]) == 0).OnlyEnforceIf(z[k,j].Not())
    #         model.AddMaxEquality(z[k,j],[x[i,j] for i in data['items'] if data['orders'][i] == k])    

    # add a constraint for each order k that the (sum of y[j,k] for all j) <= 1
    # for k in set(data['orders']):
    #     model.Add(cp_model.LinearExpr.Sum([z[k,j] for j in data['bins']]) == 1)
    
    # Change this to [z[k,j] for z in set(data['orders'])] if you will work on the item level
    for j in data['bins']:
        model.Add(cp_model.LinearExpr.Sum([x[i,j] for i in set(data['items'])]) <= data['bin_orders'][j])

    for h in set(data['clients']):
        for j in data['bins']:
            # model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items'] if data['clients'][i] == h ]) >= 1).OnlyEnforceIf(c[h,j])
            # model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items'] if data['clients'][i] == h ]) == 0).OnlyEnforceIf(c[h,j].Not())
            model.AddMaxEquality(c[h,j],[x[i,j] for i in data['items'] if data['clients'][i] == h])   

    for h in set(data['clients']):
        model.Add(cp_model.LinearExpr.Sum([c[h,j] for j in data['bins']]) == 1)
        # model.AddExactlyOne([c[h,j] for j in data['bins']])

    # # Constraint for linking the items to the subregions
    for m in set(data['subregions']):
        for j in data['bins']:
            # model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items'] if data['subregions'][i] == m ]) >= 1).OnlyEnforceIf(s[m,j])
            # model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items'] if data['subregions'][i] == m ]) == 0).OnlyEnforceIf(s[m,j].Not())
            model.AddMaxEquality(s[m,j],[x[i,j] for i in data['items'] if data['subregions'][i] == m]) 
    
    # for j in data['bins']:
    #     model.AddMaxEquality(ss[j],[x[i,j]*data['conditional_subregions'][i] for i in data['items']]) 
    
    # for m in set(data['regions']):
    #     for j in data['bins']:
    #         # model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items'] if data['subregions'][i] == m ]) >= 1).OnlyEnforceIf(s[m,j])
    #         # model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items'] if data['subregions'][i] == m ]) == 0).OnlyEnforceIf(s[m,j].Not())
    #         model.AddMaxEquality(r[m,j],[x[i,j] for i in data['items'] if data['regions'][i] == m]) 
            
    # Constraint for disallowing NKR trucks to go to the conditional subregions (kilo 4.5 and mansheyet naser)
    for j in data['bins']:
        if j in range(DABABA_COUNT + (VAN_COUNT*2),DABABA_COUNT + (VAN_COUNT*2) + (TRICYCLE_COUNT*2)):
            model.Add(cp_model.LinearExpr.Sum([x[i,j]*data['conditional_subregions'][i] for i in data['items']]) ==0)

    # # add a constraint for the number of mega-regions traversed by the same truck
    # for j in data['bins']:
    #     model.Add(cp_model.LinearExpr.Sum([r[m,j] for m in set(data['regions'])]) <= MAX_MEGA_REGIONS)  
            
    # # add a constraint for the number of regions traversed by the same truck
    # for j in data['bins']:
    #     model.Add(cp_model.LinearExpr.Sum([r[m,j] for m in set(data['regions'])]) <= MAX_REGIONS)  
    # # THIS WAS THE PROBLEM WHEN ITS <= 3
    # #add a constraint for each subregion k that the (sum of y[j,k] for all j) <= n
    # for j in data['bins']:
    #     model.Add(cp_model.LinearExpr.Sum([s[m,j] for m in set(data['subregions'])]) <= MAX_SUBREGIONS)    
    # for j in data['bins']:
    #     model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items']]) ==2).OnlyEnforceIf(o[j])
    #     model.Add(cp_model.LinearExpr.Sum([x[(i, j)] for i in data['items']]) !=2).OnlyEnforceIf(o[j].Not())
    
    for ind,m in enumerate(set(data['subregions'])):
        for n in list(set(data['subregions']))[ind:]:
            for j in data['bins']:
                # Same as Multiplication Equality
                model.AddBoolOr(s[m,j].Not(),s[n,j].Not(),d[j,m,n])
                model.AddImplication(d[j,m,n],s[m,j])
                model.AddImplication(d[j,m,n],s[n,j])
    
    for ind,m in enumerate(set(data['subregions'])):
        for n in list(set(data['subregions']))[ind:]:
            model.AddMaxEquality(e[m,n], (d[j,m,n] for j in data['bins']))
            model.Add(e[m,n]*dist_matrix[m][n] <=MAX_DISTANCE_SUBREGIONS*SCALING_FACTOR)
            # model.Add(e[m,n]*dist_matrix[m][n] <=10000)
    
        
    for j in data['bins']:
        model.AddMaxEquality(o[j],[x[i,j]*data['second_leg'][i] for i in data['items']])   
    
    # for j in data['bins']:
    #     model.AddMultiplicationEquality(t[j],y[j],o[j])

        
    model.Minimize(
    cp_model.LinearExpr.Sum([y[j]*data['bin_costs'][j] for j in data['bins']]) #*0.8 +
        + cp_model.LinearExpr.Sum([o[j]*data['bin_costs'][j] for j in data['bins']])
    # cp_model.LinearExpr.Sum([d[j,m,n]*dist_matrix[m,n] for j in data['bins'] for ind,m in enumerate(set(data['subregions'])) for n in list(set(data['subregions']))[ind:]])*0.2
    # cp_model.LinearExpr.Sum([var[j] for j in data['bins']])*0.2
              )
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    # solver.parameters.num_search_workers = 12
    solver.parameters.linearization_level=0
    solver.parameters.max_time_in_seconds = CPSAT_TIME_LIMIT
    status = solver.Solve(model)
    
    if status == cp_model.UNKNOWN or status == cp_model.INFEASIBLE:
        return pd.DataFrame()
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        num_bins = 0.
        result_dict = {}
        total_bin_items = []
        orders = orders.sort_values(['region_id','sub_region_id','order_id'])
        orders = orders.reset_index(drop=True)
        orders['product_weight'] = orders['product_weight']/SCALING_FACTOR
        orders['bin_id'] = 0
        for j in data['bins']:
            if solver.Value(y[j]) == 1:
                bin_items = []
                bin_weight = 0
                # bin_volume = 0
                bin_value = 0
                bin_sub_regions = []
                bin_sub_region_names = []
                bin_orders = []
                bin_legs = []
                for i in data['items']:
                    if solver.Value(x[i, j]) > 0:
                        bin_items.append(i)
                        bin_sub_regions.append(data['subregions'][i])
                        bin_sub_region_names.append(orders.iloc[i]['sub_region_name'])
                        # bin_orders.append(data['orders'][i])
                        bin_legs.append(data['second_leg'][i])
                        dist_sum = 0
                        for ind,s in enumerate(set(bin_sub_regions)):
                            for ss in list(set(bin_sub_regions))[ind:]:
                                dist_sum += dist_matrix[s][ss]
                        bin_weight += data['weights'][i]/SCALING_FACTOR
                        # bin_volume += data['volumes'][i]
                        bin_value += data['values'][i]
                if bin_weight > 0:
                    num_bins += 1
                    total_bin_items.append(bin_items)
                    orders.loc[orders.iloc[bin_items].index,'bin_id'] =  j
                    # result_dict[j] = orders[['order_id','product_id','product_weight','product_value']].iloc[bin_items]
                    print('Bin number', j)
                    print('  Items packed:', bin_items)
                    print('  Total weight:', bin_weight)
                    # print('  Total volume:', bin_volume)
                    print('  Total value:', bin_value)
                    print('  Second Legs Packed:', bin_legs)
                    # print('  Orders Packed:', set(bin_orders))
                    print('  Sub Regions Visited:', set(bin_sub_regions))
                    # print('  Sub Region Names:', set(bin_sub_region_names))
                    print('  Total distance travelled:', dist_sum/SCALING_FACTOR)
                    print()
        print()
        print('Number of bins used:', num_bins)
    else:
        print('INFEASIBLE SOLUTION!')
        orders = orders.reset_index(drop=True)
        orders['bin_id'] = -1
    
    return orders


def solve_vrp_problem(orders_region_df_opt,data):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    orders_region_df_opt_new = orders_region_df_opt.copy()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                        data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node in range((VAN_COUNT+TRICYCLE_COUNT+1)) and to_node in range((VAN_COUNT+TRICYCLE_COUNT+1)):
            return 100000
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # transit_callback_index = routing.RegisterTransitMatrix(data['distance_matrix'])

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Number of Orders constraint
    # Create counter for number of orders accumulated
    def counter_callback(from_index):
        """Returns 1 for any locations except depot."""
        # Convert from routing variable Index to user NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['orders'][from_node] if (from_node != 0) else 0
        # return 1 if (from_node not in list(range(data['vehicle_1']+data['vehicle_2']+1))) else 0

    counter_callback_index = routing.RegisterUnaryTransitCallback(counter_callback)
    routing.AddDimensionWithVehicleCapacity(
        counter_callback_index,
        VAN_ORDERS,  # null slack
        data['vehicle_orders'],  # maximum orders per vehicle
        True,  # start cumul to zero
        'Counter')

    counter_dimension = routing.GetDimensionOrDie('Counter')

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        VAN_WEIGHT*SCALING_FACTOR,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    for node in range(1,(VAN_COUNT+TRICYCLE_COUNT+1)):
        node_index = manager.NodeToIndex(node)
        routing.AddDisjunction([node_index], 0) # This number (was 5000) is based on the avg distance between nodes which is around 7k 
        
    # Allow to drop regular node with a cost.
    for node in range((VAN_COUNT+TRICYCLE_COUNT+1), len(data['demands'])):
        node_index = manager.NodeToIndex(node)
        counter_dimension.SlackVar(node_index).SetValue(0)
        capacity_dimension.SlackVar(node_index).SetValue(0)
        # routing.AddDisjunction([node_index], 100000000000)
        
    # # Allow to drop regular node with a cost.
    # for node in range((VAN_COUNT+TRICYCLE_COUNT+1), len(data['demands'])):
    #     node_index = manager.NodeToIndex(node)
    #     capacity_dimension.SlackVar(node_index).SetValue(0)
    #     # routing.AddDisjunction([node_index], 100000000000000)

    for vehicle in range(manager.GetNumberOfVehicles()): 
        if vehicle in range(data['vehicle_0'],data['vehicle_0']+data['vehicle_1']):
            routing.SetFixedCostOfVehicle(50000,vehicle)
        elif vehicle in range(data['vehicle_0']):
            routing.SetFixedCostOfVehicle(10000000,vehicle)
        else:
            routing.SetFixedCostOfVehicle(100000,vehicle)


    # This block adds a constraint to the number of kms traversed by each vehicle
    # It also further penalizes the arc values by multiplying them by the globalspancostcoefficient
    # coeff * (max_end - min_start) where max_end and min_start are the max and min distances across all routes
    # dimension_name = 'Distance'
    # routing.AddDimension(
    #     transit_callback_index,
    #     0,  # no slack
    #     20000,  # vehicle maximum travel distance
    #     True,  # start cumul to zero
    #     dimension_name)
    # distance_dimension = routing.GetDimensionOrDie(dimension_name)
    # distance_dimension.SetGlobalSpanCostCoefficient(1000)


    # Add Value constraint.
    def value_callback(from_index):
        """Returns the value of the node."""
        # Convert from routing variable Index to values NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['values'][from_node]

    value_callback_index = routing.RegisterUnaryTransitCallback(
        value_callback)
    routing.AddDimension(

        value_callback_index,
        0,  # null capacity slack
        1000000,  # vehicle maximum values
        True,  # start cumul to zero
        'Value')

    value_dimension = routing.GetDimensionOrDie('Value')
    # for vehicle in range(manager.GetNumberOfVehicles()): 
    #     if vehicle in range(data['vehicle_0'],data['vehicle_0']+data['vehicle_1']):
    #         # value_dimension.CumulVar(routing.End(vehicle)).RemoveInterval(0, 35000)
    #         value_dimension.SetCumulVarSoftLowerBound(routing.End(vehicle), VAN_VALUE, 10)
    #     elif vehicle in range(data['vehicle_0']): 
    #         # value_dimension.CumulVar(routing.End(vehicle)).RemoveInterval(0, 27200)
    #         value_dimension.SetCumulVarSoftLowerBound(routing.End(vehicle), DABABA_VALUE, 10)
    #     else:
    #         value_dimension.SetCumulVarSoftLowerBound(routing.End(vehicle), TRICYCLE_VALUE, 10)

    # value_dimension.SetGlobalSpanCostCoefficient(100)
        

    # second_leg_times = [j for j,s in enumerate(data['second_leg']) if s==1]
    # routing.AddSoftSameVehicleConstraint(second_leg_times, 10000)

    # exception_clients = [j for j,s in enumerate(data['conditional_subregions']) if s==1]
    # for stop in exception_clients:
    #     # This is the vehicle variable of this specific node (stop) 
    #     vehicle_var = routing.VehicleVar(manager.NodeToIndex(stop))
    #     # These are the values of the vehicles to be permitted
    #     values = list(range(data['vehicle_0']+data['vehicle_1']))
    #     # This says that "values" are the allowed vehicles to be the vehicle variable of that stop node
    #     vehicle_var.SetValues(values)
    #     # routing.solver().Add(routing.solver().MemberCt(vehicle_var, values))

    # for event_node in range(1, len(data['distance_matrix'])):
    #     event_index = manager.NodeToIndex(event_node)
    #     routing.AddDisjunction([event_index], 1000000)
    for stop in range(1,(VAN_COUNT+1)):
        # This is the vehicle variable of this specific node (stop) 
        vehicle_var = routing.VehicleVar(manager.NodeToIndex(stop))
        delta = data['vehicle_0']-1
        # These are the values of the vehicles to be permitted
        values = [stop+delta]
        # This says that "values" are the allowed vehicles to be the vehicle variable of that stop node
        vehicle_var.SetValues(values)
        # routing.solver().Add(routing.solver().MemberCt(vehicle_var, values))
        
        
    for stop in range((VAN_COUNT+1),(VAN_COUNT+TRICYCLE_COUNT+1)):
        # This is the vehicle variable of this specific node (stop) 
        vehicle_var = routing.VehicleVar(manager.NodeToIndex(stop))
        delta = data['vehicle_0']-1
        # These are the values of the vehicles to be permitted
        values = [-1,stop+delta]
        # This says that "values" are the allowed vehicles to be the vehicle variable of that stop node
        vehicle_var.SetValues(values)
        # routing.solver().Add(routing.solver().MemberCt(vehicle_var, values))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION)
    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(VRP_TIME_LIMIT)
    # search_parameters.use_full_propagation = True
    # search_parameters.log_search = True

    # routing.CloseModelWithParameters(search_parameters)

    # initial_solution = routing.ReadAssignmentFromRoutes(data['initial_solution'],
    #                                                     True)
    # print('Initial solution:')
    # distances_initial = print_solution(data, manager, routing, initial_solution)

    # Solve the problem.
    # routing.CloseModelWithParameters(search_parameters)
    # solution = routing.SolveFromAssignmentWithParameters(initial_solution,search_parameters)
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print('\n\n\n\n\n\n\n Final solution:')
        distances = print_solution(data, manager, routing, solution)
        
        vehicle_routes = {}
        vehicle_subregions = {}
        vehicles_used = 0
        orders_region_df_opt_new['bin_id_final'] = 0
        for veh in range(manager.GetNumberOfVehicles()):
            res = build_vehicle_route(manager, routing, solution,
                                                    data, veh, orders_region_df_opt_new)
            if res:
                vehicles_used += 1
                vehicle_routes[veh],vehicle_subregions[veh], orders_region_df_opt_new = res
                print(vehicle_routes[veh])
                # print(list(dict.fromkeys(vehicle_subregions[veh])) )
            else:
                vehicle_routes[veh] = res
            
        print("Number of Vehicles Used: {}".format(vehicles_used))

        
    return orders_region_df_opt_new,vehicle_routes,distances

def run_planner():
    print('Starting Planner...')
    DELIVERY_DATE = (date.today()+timedelta(days=1)).strftime("%Y-%m-%d")
    print('Date: ', DELIVERY_DATE)
    orders_df = get_delivery_orders(DELIVERY_DATE)
    print('Total number of items: ',orders_df.shape[0])
    print('Total number of orders: ',orders_df['order_id'].nunique())

    sub_region_coordinates = get_subregion_coordinates()
    print('Total number of subregions: ',sub_region_coordinates.shape[0])

    # REGION_LIST = orders_df[orders_df['warehouse_id']=='FDC421C4-1675-41E1-91CD-F50826AEECD0']['region_id'].unique()
    orders_df = orders_df[orders_df['region_id'].isin(REGION_LIST)].copy()
    print('Total number of planner items: ',orders_df.shape[0])
    print('Total number of planner orders: ',orders_df['order_id'].nunique())
    print('Total number of planner clients: ',orders_df['client_id'].nunique())
    print('Total number of NA products: ', orders_df['product_id'].isna().sum())
    print('Total number of NA coordinates: ', orders_df[(orders_df['order_latitude']==0) | (orders_df['order_longitude']==0)].shape[0])
    print('Total number of clients with non-confirmed locations: ', orders_df[orders_df['location_matched']==0]['client_id'].nunique())
    print('Total weight sum: ', orders_df['product_weight'].sum())
    print('Total values sum: ', orders_df['product_value'].sum())
    print('\n\n')
    print('Processing Orders...')
    orders_region_df, dist_matrix = process_region(orders_df,sub_region_coordinates)
    orders_region_df_init = orders_region_df.copy()
    print('Total number of planner items: ',orders_df.shape[0])
    print('Total number of planner orders: ',orders_df['order_id'].nunique())
    print('Total number of planner clients: ',orders_df['client_id'].nunique())
    print('Total number of NA coordinates: ', orders_df[(orders_df['order_latitude']==0) | (orders_df['order_longitude']==0)].shape[0])
    orders_region_df['trip_id'] = orders_region_df['trip_id'].fillna(-1)
    orders_region_df = orders_region_df.groupby(['date','region_id','sub_region_id','sub_region_name',
                                                 'client_id','order_id']).agg({'product_weight':'sum',
                                                                                'product_value':'sum',
                                                                                'second_leg':'max'}).reset_index()
    print('Creating Data dictionary...')
    data, orders_region_df,dedicated_orders = create_data_model(orders_region_df,dist_matrix)
    print('Total number of planner orders in dictionary: ', len(data['items']))
    if dedicated_orders.empty:
        print('Total number of exception clients in dictionary: ', 0)
    else:
        print('Total number of exception clients in dictionary: ', dedicated_orders['client_id'].nunique())
    print('\n\n')
    print('Starting CP-SAT Model....')
    orders_region_df_opt = design_model(orders_region_df,dist_matrix,data,fallback=True)

    print('\n\n')
    print('Creating VRP Data dictionary...')
    data = create_data_model_routing(orders_df,orders_region_df_opt,sub_region_coordinates)
    print('Total number of planner orders in dictionary: ', len(data['demands']))
    print('Distance matrix dimensions: ', np.array(data['distance_matrix']).shape)
    print('\n\n')
    print('Starting VRP Model....')
    orders_region_df_opt_final,vehicle_routes,distances = solve_vrp_problem(orders_region_df_opt,data)

    print(orders_region_df_opt_final.groupby(['bin_id_final']).sum().sum())

    folium_df = orders_region_df_opt_final.merge(orders_region_df_init.groupby('client_id')[['order_latitude','order_longitude']].max().reset_index(),on='client_id',how='left')
    map = draw_map(folium_df,vehicle_routes)

    if dedicated_orders.empty==False:
        dedicated_orders = dedicated_orders[[col for col in orders_region_df_opt_final.columns if col not in ['bin_id','bin_id_final']]].copy()
        if dedicated_orders['client_id'].nunique() == 1:
            dedicated_orders['bin_id'] = orders_region_df_opt_final['bin_id'].max() + 1
            dedicated_orders['bin_id_final'] = orders_region_df_opt_final['bin_id_final'].max() + 1
        else:
            dedicated_orders = dedicated_orders.sort_values(['client_id','order_id'])
            client_sequence = dedicated_orders.groupby('client_id',sort=False).ngroup().values.tolist()
            dedicated_orders['bin_id'] = client_sequence
            dedicated_orders['bin_id'] = dedicated_orders['bin_id'] + orders_region_df_opt_final['bin_id'].max() + 1
            dedicated_orders['bin_id_final'] = client_sequence
            dedicated_orders['bin_id_final'] = dedicated_orders['bin_id_final'] + orders_region_df_opt_final['bin_id_final'].max() + 1
        for bin in dedicated_orders['bin_id_final'].unique():   
            vehicle_routes[bin] = dedicated_orders[dedicated_orders['bin_id_final']==bin]['client_id'].unique().tolist()
        dedicated_orders = dedicated_orders[[col for col in orders_region_df_opt_final.columns]].copy()
        dedicated_orders['product_weight'] = dedicated_orders['product_weight']/SCALING_FACTOR
        orders_region_df_opt_final = pd.concat([orders_region_df_opt_final,dedicated_orders])


    products_df = get_product_details()
    final_df = orders_region_df_opt_final.merge(orders_df[['product_id','order_id','product_quantity','order_number','product_weight','product_value']].drop_duplicates(), on = 'order_id')
    final_df = final_df.rename(columns = {'product_weight_x':'order_weight','product_value_x':'order_value',
                                            'product_weight_y':'product_weight','product_value_y':'product_value'})
    final_df = final_df.merge(products_df[['product_id','product_number','product_name',
                                 'category','brand','case_content_description']],
                    on='product_id',
                    how = 'left')  
    final_df = final_df.sort_values(['bin_id_final','order_id'])
    final_df['mandoob_number'] = final_df.sort_values(['bin_id_final','order_id']).groupby('bin_id_final',sort=False).ngroup().values.tolist()
    final_df['mandoob_number'] = final_df['mandoob_number'] + 1
    final_df['second_run'] = np.where(final_df['bin_id_final']>=data['num_vehicles'],1,0)
    final_df['run_number'] = np.where(final_df['bin_id_final']>=data['num_vehicles'],2,1)
    final_df['first_run_bin'] = np.where(final_df['second_run']==1,final_df['bin_id_final']-data['num_vehicles'],-1)
    final_df = final_df.merge(final_df[['bin_id_final','mandoob_number']].drop_duplicates(),left_on='first_run_bin',right_on='bin_id_final',how='left')
    final_df = final_df.drop(columns = {'bin_id_final_y'}).rename(columns = {'bin_id_final_x':'bin_id_final'})
    final_df['mandoob_number'] = np.where(final_df['mandoob_number_y'].isna()==False,final_df['mandoob_number_y'],final_df['mandoob_number_x'])
    final_df['mandoob_number'] = final_df['mandoob_number'].astype(int)
    final_df['mandoob'] = 'مندوب ' +  final_df['mandoob_number'].astype(str) + '-' +final_df['run_number'].astype(str)

    final_df = final_df.merge(final_df.groupby('bin_id_final')['order_id'].nunique().to_frame().reset_index(), on='bin_id_final')    
    final_df = final_df.rename(columns = {'order_id_y':'order_count','order_id_x':'order_id'})   

    vehicle_distances = pd.DataFrame(zip(final_df['bin_id_final'].sort_values().unique().tolist(),[x for x in distances if x != 0]))
    vehicle_distances.columns = ['bin_id_final','distance_travelled']      
    final_df = final_df.merge(vehicle_distances, on='bin_id_final', how='left')    
    final_df['distance_travelled'] = final_df['distance_travelled'].fillna(0)

    print('Final weight and value:')
    print(final_df.groupby('bin_id_final')[['product_weight','product_value']].sum().sum())

    trucks = final_df['bin_id_final'].unique().tolist()         
    
    delivery_date_formated = pd.to_datetime(DELIVERY_DATE).strftime('%Y%m%d')
    PATH = os.path.join(settings.STATIC_ROOT,'warehouse_optimization_orders/{}'.format(delivery_date_formated))
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    with pd.ExcelWriter('{}/area_runsheet_{}.xlsx'.format(PATH,DELIVERY_DATE)) as writer:
        for truck in trucks:
            mandoob = final_df[final_df['bin_id_final'] == truck]['mandoob'].iloc[0]
            single_truck = final_df[final_df['bin_id_final'] == truck][['product_name','product_quantity','case_content_description','product_value','product_weight','bin_id_final','mandoob','order_count']].copy()
            single_truck.loc['Total']= single_truck[['product_weight','product_value']].sum()
            single_truck.loc['subregions','product_name'] = ",".join(final_df[final_df['bin_id_final'] == truck]['sub_region_name'].unique().tolist())
            single_truck.loc['second_leg_count','order_count'] = final_df[final_df['bin_id_final']==truck].groupby(['order_id'])['second_leg'].max().sum()
            single_truck.to_excel(writer, 
                                index=True, 
                                sheet_name=mandoob)

    for truck in trucks:
        orders = final_df.loc[final_df['bin_id_final'] == truck,['client_id','order_id','order_number']].drop_duplicates()
        if dedicated_orders.empty == False:
            dedicated_clients = orders['client_id'].isin(dedicated_orders['client_id'].values).sum()
        else:
            dedicated_clients = 0
        if truck >= data['num_vehicles'] and dedicated_clients == 0:
            truck_init = truck - data['num_vehicles']
            return_points = [i for i, v in enumerate(vehicle_routes[truck_init]) if v == 'warehouse' and i not in [0,len(vehicle_routes[truck_init])-1]]
            client_ids_order = vehicle_routes[truck_init][return_points[0]:]
        else:    
            client_ids_order = vehicle_routes[truck]
        custom_dict = dict(zip(client_ids_order,range(len(client_ids_order))))
        orders = orders.sort_values('client_id',key=lambda x: x.map(custom_dict))
        with pd.ExcelWriter('{}/area_vehicle_{}_orders_{}.xlsx'.format(PATH,truck,DELIVERY_DATE)) as writer:
            orders[['order_id','order_number']].to_excel(writer, index=None)

    map.save('{}/area_map_{}.html'.format(PATH,DELIVERY_DATE))

    return final_df

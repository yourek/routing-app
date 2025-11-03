import math
import numpy as np
import igraph as ig
import osmnx as ox
import networkx as nx
from pathlib import Path
import streamlit as st
    
def closest_reachable_node(G, source, target, pos_attr='pos', consider_directed=True, turn_penalty=6):
    """
    Returns a node reachable from source that minimises euclidean 
    distance to target, plus path length from source to that node.
    """
    try:
        # first try to get exact path
        #length = nx.shortest_path_length(G, source, target, weight='travel_time')

        #path = nx.shortest_path(
        #        G, 
        #        source, 
        #        target, 
        #        weight="travel_time"
        #    )
        
        length, path = nx.bidirectional_dijkstra(
            G, 
            source, 
            target, 
            weight="travel_time"
            )
        
        length += (len(path)-2)*turn_penalty

        return length
    except nx.NetworkXNoPath:
        return np.nan
    
def PrecalculateDistancesFaster(name, roles):

    import json 
    import pickle

    import geopandas as gpd
    from shapely.geometry import Point
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import cKDTree

    # We’ll build points as (lon, lat) so it lines up with node['x'], node['y']
    points = np.array([
        (r['Longitude'], r['Latitude']) 
        for r in roles
    ])

    # 2. Load your pre‐pickled OSMnx Graph

    Path("assets/Time_In_Store_Estimation/G_drives").mkdir(parents=True, exist_ok=True)

    with open(
        'assets/Time_In_Store_Estimation/G_drives/pickle_G_drive_%s.pkl' % name.rsplit('-', 1)[0],
        'rb'
    ) as f:
        G_drive = pickle.load(f)

    # 3. Extract node‐ids and their (x,y) coords into parallel arrays
    nodes = list(G_drive.nodes())
    coords = np.array([
        (G_drive.nodes[n]['x'], G_drive.nodes[n]['y']) 
        for n in nodes
    ])

    # 4. Build a cKDTree on the coords for super‐fast nearest‐neighbour lookups
    tree = cKDTree(coords)

    # 5. Query every point, in batch or with a progress bar
    # Batch (fastest):
    _, idxs = tree.query(points, k=1)
    nearest_nodes = [nodes[i] for i in idxs]

    nearest_nodes = []
    for pt in points:
        _, i = tree.query(pt, k=1)
        nearest_nodes.append(nodes[i])

    n = len(nearest_nodes)
    D = [[0]*n for _ in range(n)]

    points = gpd.GeoSeries([Point(xy) for xy in points])
    xy_array = np.array([[point.x, point.y] for point in points])
    D = squareform(pdist(xy_array, metric='euclidean'))

    Path("assets/Time_In_Store_Estimation/Distances").mkdir(parents=True, exist_ok=True)

    with open('assets/Time_In_Store_Estimation/Distances/D_%s.pkl' % name.rsplit('-', 1)[0], 'wb') as f:
        pickle.dump((D.tolist(), nearest_nodes), f)

    return 0

import math
import pickle
import random
import numpy as np
from itertools import permutations
from functools import lru_cache

def load_distance_matrix(name, attendance_mask):
    Path("assets/Time_In_Store_Estimation/Distances").mkdir(parents=True, exist_ok=True)
    with open(f'assets/Time_In_Store_Estimation/Distances/D_{name.rsplit("-", 1)[0]}.pkl', 'rb') as f:
        D_full, _ = pickle.load(f)
    D = np.array(D_full)
    mask = attendance_mask == 1
    return D[mask][:, mask]

def tour_length(tour, D):
    return sum(D[tour[i], tour[(i+1) % len(tour)]] 
               for i in range(len(tour)))

def nearest_neighbor(D, start):
    n = D.shape[0]
    unvisited = set(range(n)) - {start}
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: D[last, j])
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour

def cheapest_insertion(D):
    n = D.shape[0]
    # seed with the closest pair
    i, j = max(((i, j) for i in range(n) for j in range(i+1, n)),
               key=lambda ij: D[ij[0], ij[1]])
    tour = [i, j]
    remaining = set(range(n)) - set(tour)
    while remaining:
        best = None
        delta = math.inf
        for v in remaining:
            for pos in range(len(tour)):
                w = tour[(pos+1) % len(tour)]
                inc = D[tour[pos], v] + D[v, w] - D[tour[pos], w]
                if inc < delta:
                    delta, best = inc, (v, pos+1)
        tour.insert(best[1], best[0])
        remaining.remove(best[0])
    return tour

def two_opt(tour, D):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n if i>0 else n-1):
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%n]
                gain = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if gain < -1e-9:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
        # repeat until no improvement
    return tour

def held_karp(D):
    n = len(D)
    # DP[(mask, last)] = cost to reach 'last' having visited 'mask'
    DP = {}
    @lru_cache(None)
    def visit(mask, last):
        if mask == (1 << n) - 1:
            return D[last, 0]
        best = math.inf
        for nxt in range(n):
            if not (mask & (1 << nxt)):
                best = min(best,
                           D[last, nxt] + visit(mask | (1 << nxt), nxt))
        return best
    best_cost = math.inf
    best_end = None
    # try all endpoints
    for end in range(n):
        cost = D[0, end] + visit(1<<0 | 1<<end, end)
        if cost < best_cost:
            best_cost, best_end = cost, end
    # no easy way to reconstruct path here without backtracking code,
    # so we just return cost for small n
    return None, best_cost

def FindBestOrderImproved(name, attendance_mask,
                  method='cheapest_insertion',
                  local_improve=True,
                  runs=5,
                  brute_force_threshold=7):
    """
    method: 'nearest_neighbor' | 'cheapest_insertion'
    local_improve: apply 2-Opt if True
    runs: how many random restarts (for NN only)
    brute_force_threshold: n <= threshold → brute force by Held-Karp
    """
    D = load_distance_matrix(name, attendance_mask)
    n = D.shape[0]

    # for very small n, use exact DP
    #if n <= brute_force_threshold:
    #    _, cost = held_karp(D)
    #    # reconstructing exact tour is omitted for brevity
    #    return None, cost

    best_cost = math.inf
    best_tour = None

    for _ in range(runs if method=='nearest_neighbor' else 1):
        # 1) Constructive phase
        if method == 'nearest_neighbor':
            #start = random.randrange(n)
            start = np.argmax((D**2).sum(axis=0))
            tour = nearest_neighbor(D, start)
        else:
            tour = cheapest_insertion(D)

        # 2) Local improvement
        if local_improve:
            tour = two_opt(tour, D)

        cost = tour_length(tour, D)
        if cost < best_cost:
            best_cost, best_tour = cost, tour

    return best_tour, None

def FindRealRouteTime(name, order, mask, turn_penalty_seconds=6):
    import pickle
    import numpy as np

    #conservative_multiplier = {5*60:2, 10*60:1.2, 30*60:1.15, 60*60:1.1, np.inf:1}
    #conservative_multiplier = {5*60:3, 10*60:2, 30*60:1.5, 60*60:1.2, np.inf:1}

    Path("assets/Time_In_Store_Estimation/Distances").mkdir(parents=True, exist_ok=True)
    Path("assets/Time_In_Store_Estimation/G_drives").mkdir(parents=True, exist_ok=True)

    with open('assets/Time_In_Store_Estimation/Distances/D_%s.pkl' % name.rsplit('-', 1)[0], 'rb') as f:
        _, nodes = pickle.load(f)
    
    nodes = [x for i, x in enumerate(nodes) if mask[i]==1]

    with open('assets/Time_In_Store_Estimation/G_drives/pickle_G_drive_%s.pkl' % name.rsplit('-', 1)[0], 'rb') as f:
        G_drive = pickle.load(f)

    pos = {}
    for node, data in G_drive.nodes(data=True):
        # try OSM default x/y, else lon/lat
        x = data.get('x', data.get('lon'))
        y = data.get('y', data.get('lat'))
        if x is None or y is None:
            raise KeyError(f"Node {node} missing coordinate keys")
        pos[node] = (x, y)

    # Set it all at once
    nx.set_node_attributes(G_drive, pos, 'pos')

    times = []

    for start, end in zip(order[:-1], order[1:]):

        time = closest_reachable_node(G_drive, nodes[start], nodes[end], turn_penalty=turn_penalty_seconds)

        times.append(time)

    times = np.where(np.isnan(times), np.nanmean(times), times).tolist()
        
    return times

def detect_required_countries(stores_json):

  import json
  from shapely import Point
  import geopandas as gpd
  import glob

  roles = stores_json

  Path("assets/Time_In_Store_Estimation").mkdir(parents=True, exist_ok=True)

  gdf = gpd.read_file('assets/Time_In_Store_Estimation/World_Countries_Generalized_Shapefile').to_crs('EPSG:4326')

  mapping = {
    "Algeria": "algeria-251005.osm.pbf",
    "Angola": "angola-251005.osm.pbf",
    "Armenia": "armenia-251005.osm.pbf",
    "Azerbaijan": "azerbaijan-251005.osm.pbf",
    "Benin": "benin-251005.osm.pbf",
    "Botswana": "botswana-251005.osm.pbf",
    "Burkina Faso": "burkina-faso-251005.osm.pbf",
    "Burundi": "burundi-251005.osm.pbf",
    "Cameroon": "cameroon-251005.osm.pbf",
    "Canarias": "canary-islands-251005.osm.pbf",
    "Cabo Verde": "cape-verde-251005.osm.pbf",
    "Central African Republic": "central-african-republic-251005.osm.pbf",
    "Chad": "chad-251005.osm.pbf",
    "Comoros": "comores-251005.osm.pbf",
    "Congo": "congo-brazzaville-251005.osm.pbf",
    "Congo DRC": "congo-democratic-republic-251005.osm.pbf",
    "Djibouti": "djibouti-251005.osm.pbf",
    "Egypt": "egypt-251005.osm.pbf",
    "Equatorial Guinea": "equatorial-guinea-251005.osm.pbf",
    "Eritrea": "eritrea-251005.osm.pbf",
    "Ethiopia": "ethiopia-251005.osm.pbf",
    "Gabon": "gabon-251005.osm.pbf",
    "Ghana": "ghana-251005.osm.pbf",
    "Guinea": "guinea-251005.osm.pbf",
    "Guinea-Bissau": "guinea-bissau-251005.osm.pbf",
    "Iran": "iran-251005.osm.pbf",
    "Iraq": "iraq-251005.osm.pbf",
    "Côte d'Ivoire": "ivory-coast-251005.osm.pbf",
    "Jordan": "jordan-251005.osm.pbf",
    "Kazakhstan": "kazakhstan-251005.osm.pbf",
    "Kenya": "kenya-251005.osm.pbf",
    "Kyrgyzstan": "kyrgyzstan-251005.osm.pbf",
    "Lebanon": "lebanon-251005.osm.pbf",
    "Lesotho": "lesotho-251005.osm.pbf",
    "Liberia": "liberia-251005.osm.pbf",
    "Libya": "libya-251005.osm.pbf",
    "Madagascar": "madagascar-251005.osm.pbf",
    "Malawi": "malawi-251005.osm.pbf",
    "Mali": "mali-251005.osm.pbf",
    "Mauritania": "mauritania-251005.osm.pbf",
    "Mauritius": "mauritius-251005.osm.pbf",
    "Morocco": "morocco-251005.osm.pbf",
    "Mozambique": "mozambique-251005.osm.pbf",
    "Namibia": "namibia-251005.osm.pbf",
    "Niger": "niger-251005.osm.pbf",
    "Nigeria": "nigeria-251005.osm.pbf",
    "Pakistan": "pakistan-251005.osm.pbf",
    "Rwanda": "rwanda-251005.osm.pbf",
    "Saint Helena": "saint-helena-ascension-and-tristan-da-cunha-251005.osm.pbf",
    "Sao Tome and Principe": "sao-tome-and-principe-251005.osm.pbf",
    "Senegal": "senegal-and-gambia-251005.osm.pbf",
    "Seychelles": "seychelles-251005.osm.pbf",
    "Sierra Leone": "sierra-leone-251005.osm.pbf",
    "Somalia": "somalia-251005.osm.pbf",
    "South Africa": "south-africa-and-lesotho-251005.osm.pbf",
    "South Sudan": "south-sudan-251005.osm.pbf",
    "Sudan": "sudan-251005.osm.pbf",
    "Eswatini": "swaziland-251005.osm.pbf",
    "Syria": "syria-251005.osm.pbf",
    "Tajikistan": "tajikistan-251005.osm.pbf",
    "Tanzania": "tanzania-251005.osm.pbf",
    "Togo": "togo-251005.osm.pbf",
    "Tunisia": "tunisia-251005.osm.pbf",
    "Turkmenistan": "turkmenistan-251005.osm.pbf",
    "Uganda": "uganda-251005.osm.pbf",
    "Uzbekistan": "uzbekistan-251005.osm.pbf",
    "Zambia": "zambia-251005.osm.pbf",
    "Zimbabwe": "zimbabwe-251005.osm.pbf",
    "Israel": "israel-and-palestine-250930.osm.pbf",
    "Palestinian Territory": "israel-and-palestine-250930.osm.pbf",
    'United Arab Emirates': "gcc-states-251005.osm.pbf",
    'Oman':"gcc-states-251005.osm.pbf",
    'Kuwait':"gcc-states-251005.osm.pbf",
    'Bahrain':"gcc-states-251005.osm.pbf",
    'Qatar':"gcc-states-251005.osm.pbf"
  }

    # finding which countries have their pickled shapefile in folder
  mapping = {k:mapping[k] for k in mapping if mapping[k].rsplit('-', 1)[0] in [
      z.rsplit('drive_', 1)[1].rsplit('.', 1)[0] for z in glob.glob("assets/Time_In_Store_Estimation/G_drives/*.pkl")]}

  countries_cusotmers = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=[Point(x['Longitude'],x['Latitude']) for x in roles], 
                                                           data = {'Customer':[x['Customer'] for x in roles]},
                                          crs='EPSG:4326'), gdf, how='left', max_distance=0.05)
  
  countries_cusotmers = countries_cusotmers[['Customer', 'geometry', 'COUNTRY']]

  countries_cusotmers['file'] = countries_cusotmers['COUNTRY'].map(mapping)

  unmapped_countries = countries_cusotmers.loc[countries_cusotmers['file'].isna()]['COUNTRY'].unique()

  if unmapped_countries.shape[0]>0:
    st.warning("Countries of " + ', '.join(unmapped_countries.tolist()
        ) + ' are not having their roadmaps loaded, you need to add roadmaps or estimate values manualy', 
                icon="⚠️")
  
  countries = countries_cusotmers['COUNTRY'].unique()

  files = set([mapping.get(x, None) for x in countries])
  files = [x for x in files if x is not None]

  return files, countries_cusotmers, unmapped_countries

def TimeInStoreEst(stores_json, min_cluster_size=300, clusters_sampled=6, lunch_brake_in_minutes=30, in_out_penalty_minutes=5, turn_penalty_seconds=6):

    import math
    import random
    import json
    import numpy as np
    import numpy as np
    from k_means_constrained import KMeansConstrained

    roles_all_countries = stores_json

    road_file_names, customer_mapping, unmapped_countries = detect_required_countries(roles_all_countries)
    results_countries = {}

    if len(road_file_names)==0:
        return None, unmapped_countries

    for file in road_file_names:

        customer_mapping_country_selected = customer_mapping.loc[customer_mapping['file']==file]
        roles = [y for y in roles_all_countries if y['Customer'] in customer_mapping_country_selected['Customer'].unique()]

        PrecalculateDistancesFaster(file, roles)

        X = np.array([(x['Latitude'], x['Longitude']) for x in roles])

        if min_cluster_size > len(X):
            min_cluster_size = len(X)

        k = len(X) // min_cluster_size

        if clusters_sampled > k:
            clusters_sampled = k

        model = KMeansConstrained(
            n_clusters=k,
            size_min=min_cluster_size,
            size_max=len(X),  
            random_state=42
        )
        model.fit(X)
        labels = model.labels_
        roles = [x|{'cluster':int(labels[i])} for i, x in enumerate(roles)]

        all_clusters = set([x['cluster'] for x in roles])

        def checker(x):
            if (type(x) is str) or (x is None) or (type(x) is bool):
                return False
            if x > 0:
                return True
            else:
                return False
        
        attendence_visits = [{y.split(' -')[0] for y in x if ('visits per month' in y) & checker(x[y])} for x in roles]

        attendence_time = [{y.split(' -')[0] for y in x if ('time in store' in y) & checker(x[y])} for x in roles]

        attendence = [s1 & s2 for s1, s2, in zip(attendence_visits, attendence_time)]

        all_roles = list(set.union(*attendence))

        mask = [[0]*len(all_roles) for x in roles]
        mask = np.zeros((len(all_roles), len(roles), len(all_clusters)))

        for i, rep_class in enumerate(all_roles):
            for j, r in enumerate(roles):
                for ji, c in enumerate(all_clusters):
                    mask[i, j, ji] = (rep_class in attendence[j]) & (r['cluster']==c)

        res = {}
        for i, rep_class in enumerate(all_roles):

            roles_rep_class = [x for j, x in enumerate(roles) if np.any(mask[i, j, :]==1)]
            all_clusters = set([x['cluster'] for x in roles_rep_class])
            sampled_clusters = random.sample(list(all_clusters), clusters_sampled)

            clusters_res = []

            for cluster in sampled_clusters:

                roles_rep_class_cluster = [x for x in roles_rep_class if x['cluster']==cluster]
            
                semi = [{y:x[y] for y in x if rep_class in y} for x in roles_rep_class_cluster]

                time_in_stores = sum(
                    [
                        x[[y for y in x if 'time in store' in y][0]]/x[
                    [y for y in x if 'visits per month' in y][0]] for x in semi
                    ]
                )

                if mask[i, :, cluster].sum() == 1:
                    continue

                best_order, best_cost = FindBestOrderImproved(file, mask[i, :, cluster])

                if best_order is not None:

                    best_cost = sum(FindRealRouteTime(file, best_order, mask[i, :, cluster], turn_penalty_seconds=turn_penalty_seconds))
                    best_cost += in_out_penalty_minutes*2*60
                    best_cost_hours = best_cost / 3600

                    res_without_lunch_brake = time_in_stores/(time_in_stores+best_cost_hours)
                    lunch_fraction = lunch_brake_in_minutes / (8*60)
                    new_work_time = (1-lunch_fraction)*res_without_lunch_brake
                    new_drive_time = (1-lunch_fraction)*(1-res_without_lunch_brake) + lunch_fraction
                    res_wit_lunch_brake = new_work_time / (new_work_time+new_drive_time)

                    clusters_res.append(res_wit_lunch_brake)

            res[rep_class] = np.mean(clusters_res)

        results_countries[file] = res
    return results_countries, unmapped_countries

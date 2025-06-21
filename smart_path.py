import heapq
import math
import requests
import os

# Each node is a point on the map
nodes = {
    'A': (0, 0),
    'B': (1, 0),
    'C': (1, 1),
    'D': (0, 1),
    'E': (2, 1)
}

# edge has: (destination, distance, danger score)
graph = {
    'A': [('B', 1, 0.1), ('D', 1, 0.2)],
    'B': [('A', 1, 0.1), ('C', 1, 0.5)],
    'C': [('B', 1, 0.5), ('E', 1.5, 0.1)],
    'D': [('A', 1, 0.2), ('C', 1, 0.3)],
    'E': [('C', 1.5, 0.1)]
}


def heuristic(a, b):
    # Straight-line distance (Euclidean)
    return math.dist(nodes[a], nodes[b])

def a_star_safest(start, goal):
    open_set = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor, dist, danger in graph.get(current, []):
            danger_penalty = danger * 10  # Tune this weight
            new_cost = cost_so_far[current] + dist + danger_penalty

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    return None  # No path found


import requests

def get_traffic_delay(origin, destination, api_key):
    url = (
        "https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={origin}&destination={destination}&departure_time=now&key={api_key}"
    )
    res = requests.get(url).json()
    try:
        leg = res['routes'][0]['legs'][0]
        base = leg['duration']['value']  # Normal duration (seconds)
        traffic = leg['duration_in_traffic']['value']  # With traffic (seconds)
        delay = traffic - base
        return delay / 300.0  # Normalize delay to 0–1 scale (5 min max delay = 1.0)
    except Exception as e:
        print("Error:", e)
        return 0.5  # fallback risk value


api_key = os.getenv("API_KEY")
print(api_key)

# how to compute safety score to influence the A*
# safety_score = w1 * crime_risk + w2 * traffic_risk + w3 * environmental_risk + w4 * CV_hazard_score


print(get_traffic_delay("15720 Kelbaugh Rd", "1709 Shattuck Ave",api_key))

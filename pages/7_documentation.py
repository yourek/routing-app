import streamlit as st

from session.session import init_session
from utils.auth_utils.guards import require_authentication

init_session()
require_authentication()


st.title("📚 Documentation of a model")

st.markdown(
    """
## Step 1: Regionalization
### Objective
Assign stores to sales representatives in a way that balances workload while considering both working and travel time.

### Inputs
- **Store data:**
    - IDs
    - Location
- **Parameters:** 
    - Working weekdays
    - Workday time limits
    - Idles (including lunch breaks, average parking time and possibly others)
    - Optional: time in store factor - measure of how much time in store single FTE can spend across working hours. If not provided, it will be calculated automatically
- **Roles**
    - Role names
    - Visits number per month (either 0, 1, 2, 4, 8, 12, 16 or 20)
    - Monthly time in store

### Approach
1. Number of FTEs are calculated automatically per each role. You can manually increase the numbers in the UI.
2. If time in store factor is not given by the user (left as default 100) engine calculates it using OpenStreetMap API.
3. Clustering is done with **Weighted Constrained K-Means** model for each role separately.
    1. Each store has weight assigned based on role monthly time in store. For example, if a site requires 4 hours per month, its data point will be duplicated 4 times in the K-Means algorithm.
    2. K-Means algorithm performs clustering, not to exceed total monthly FTE capacity.
    3. Each cluster corresponds to one sales representative.
4. In the application you can manually change the store - sales representative assignment after review.

**Important:** 
- The algorithm does not guarantee that each cluster will remain within the target limit per region. In extreme cases, where a site has a high workload (e.g., 50 hours per month) and is located near the boundary of a region, K-Means may split the duplicated points across two clusters (e.g., 30 hours assigned to one cluster and 20 hours to another).
- To resolve this, the site is ultimately assigned to the cluster with the larger share of points. This ensures each site belongs to exactly one cluster. However, this adjustment can disrupt workload balance: removing 20 hours from one cluster and adding them to another may cause the latter to exceed its capacity.

In principle, each cluster is designed to be both:
- **Workload balanced** (no representative is over capacity).
- **Geographically coherent** (reduces unnecessary travel).


### Outputs
- .json files with store - sales representative assignment, separate one for each role
- sales representative workload visualization
- map with territories covered by sales representatives

### Assumptions
- The clustering is done on 4-week schedule, which is equivalent to 20 working days.

---

## Step 2: Week Distribution
### Objective
Spread visits across the month for each role, assigning the visits to weeks and potential days.

### Inputs
- Clustering results
- Visits number per month for each role
- **Parameters:** 
    - Working weekdays
    - Workday time limits
    - Information if delivery dates should be taken into account in the optimization
    - Visits Window for each role - how many days before / after the delivery sales representative visits should happen

### Approach
1. **Explode store data into individual visits** based on the visit number column.
- Handle cases where visits occur multiple times per week by assigning specific weekdays.
- Prepare weekly visit schedules (straightforward case).
- For biweekly and monthly visits, apply a **mini version of Weighted Constrained K-Means clustering** to ensure that, for each week, visit clusters are balanced and require roughly the same amount of time.
2. Assign potential days when sales representatives should visit the stores based on delivery dates and visits windows

### Outputs
- .json files with week distribution - store visits spread, separate file for each role
- weekly distribution visualization - workload per week and per sales representative

---

## Step 3: Route Optimization
### Objective
Determine the most efficient sequence of store visits for each representative.

### Inputs
- **Store data:**
    - IDs
    - Location
- Week distribution output
- **Parameters:** 
    - Working weekdays
    - Workday time limits
    - Idles (including lunch breaks, average parking time and possibly others)
    - Information if delivery dates should be taken into account in the optimization
    - Information if optimization engine should account for road traffic

### Approach
1. **For each role, week, and sales representative separately:**
- Apply a **route optimisation algorithm** using [Google Maps Route Optimization](https://developers.google.com/maps/documentation/route-optimization)
- Each day is treated as a separate "vehicle" for the optimization engine
2. **Standardise Google Maps responses** into a common tabular format.
3. Save visits that are skipped due to the constraints.

### Outputs
- .json files with optimized schedule for sales representatives, including visit, travel and idles time - separate file for each role
- .json files with visits skipped due to the constraints
- average time allocation per visit and role visualization
- performance KPIs per role
- summary of unallocated visits

### Assumptions
- All calculations and optimisations are based on a full-time employee model, no part time engagements are allowed.

### Generating Route Optimization prerequisites

Route Optimization process uses **external Google API**. To be able run this step, you need to allow using it and create a file with Google Cloud authorization credentials. Below you can find instructions on how to generate it.

- In case you do not have Google account, create it.
- Go to [Google Cloud Console](https://console.cloud.google.com).
- Click "Create new project".
- Fill "Project name" section.
- Click "Create" button.
- Navigate to APIs and services -> Enabled APIs and services.
- Search for "Route Optimization API".
- Click "Enable".
- Navigate to billing.
- Create a billing account if you don't have any, providing proper debit card details.
- Link a billing account to the project - keep in mind that the card will be charged for cost of using Route Optimization API.
- Navigate to IAM and admin -> Service accounts.
- Click "Create service account".
- Fill "Service account name" section.
- Click "Create and continue".
- Add role "Owner" in "Permissions" section.
- Click "Continue" and "Done".
- Click on newly created service account.
- Go to "Keys" tab.
- Click Add key -> Create new key -> JSON -> Create.
- The .json file with authorization credentials will be saved to your PC.

"""
)

import os

target_dir = "./config"
os.makedirs(target_dir, exist_ok=True)
target_path = os.path.join(target_dir, "application_default_credentials.json")

# File uploader widget
uploaded_file = st.file_uploader(
    "Upload your application_default_credentials.json", type=["json"]
)

if uploaded_file is not None:
    # Save file to the target location
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Credentials file successfully saved to: {target_path}")

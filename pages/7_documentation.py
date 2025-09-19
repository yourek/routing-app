import streamlit as st

st.title("📚 Documentation of a model")

st.markdown(
    """
## Step 1: Store Assignment via Weighted K-Means Clustering
### Objective
Assign stores to sales representatives in a way that balances workload while considering both working time and travel time.

### Inputs
- **Store data:** location, delivery dates.
- **Parameters:** 
    - Max monthly working capacity e.g. 170 hours
    - Target workload per region: ~83% of monthly capacity (e.g. ~141 hours)
    - Workday limits are 8:30 - 17:30 (including 30 minutes lunch break, so effectively max 8.5 hours of work per day)

### Approach
1. Choose the number of clusters such that the average workload per cluster is ~83% of a 170 hours representative’s monthly capacity (~141 hours).
2. Apply **Weighted Constrained K-Means clustering** to group stores into potential sales representative territories.
3. Assign weights based on monthly on-site visit duration. For example, if a site requires 4 hours per month, its data point will be duplicated 4 times in the K-Means algorithm.
1. **Important:** 
    1. The algorithm does not guarantee that each cluster will remain within the target limit of ~141 hours per region. In extreme cases (especially merchandising), where a site has a high workload (e.g., 50 hours per month) and is located near the boundary of a region, K-Means may split the duplicated points across two clusters (e.g., 30 hours assigned to one cluster and 20 hours to another).
    2. To resolve this, the site is ultimately assigned to the cluster with the larger share of points. This ensures each site belongs to exactly one cluster. However, this adjustment can disrupt workload balance: removing 20 hours from one cluster and adding them to another may cause the latter to exceed its capacity.
4. In principle, each cluster is designed to be both:
    - **Workload balanced** (no representative is over capacity).
    - **Geographically coherent** (reduces unnecessary travel).

### Outputs
- **clusters_map.html** and **all_clusers_map.html** – Territory maps per representative.
- **stores_group.xlsx** – Store data with assigned clusters / regions.
- This Excel can be adjusted by hand, if switching some region assignments makes more sense.

### Assumptions
- All visit durations are adjusted from a 22-day month to a 20-day month in order to align with a 4-week model. For this adjustment, hours are multiplied by ~91% (20/22).

---

## Step 2: Route Optimization within Assigned Territories
### Objective
Determine the most efficient sequence of site visits for each representative’s assigned territory.

### Inputs
- **stores_group.xlsx** – Store data with assigned clusters / regions, generated in Step 1.
- **Parameters:** 
- Additional buffer: 5 minutes parking/walking per visit (10 minutes in Tel Aviv).

### Approach
1. **Explode store data into individual visits** based on the visit frequency column.
- Handle cases where visits occur multiple times per week by assigning specific weekdays.
- Prepare weekly visit schedules (straightforward case).
- For biweekly and monthly visits, apply a **mini version of Weighted Constrained K-Means clustering** to ensure that, for each week, visit clusters are balanced and require roughly the same amount of time.
2. **Incorporate distribution day constraints** into visit windows if selected (discouraged, as these constraints significantly increase complexity and risk of not finding acceptable solution).
3. **For each week, per representative:**
- Apply a **route optimisation algorithm** (Google Maps Route Optimization – Fleet Routing).
- Each "vehicle" in the "fleet" is treated as a different weekday for a given week and representative.
4. **Standardise Google Maps responses** into a common tabular format.
- Add an additional parking/walking buffer to results. This may extend some workdays beyond 8.5 hours but can be adjusted, removed, or ignored as needed.
- Calculate total **work + travel time** per representative.
5. **Validate feasibility:**
- Ensure total time is approximately equal to monthly capacity.
- Manually adjust schedules if the optimisation algorithm skips certain visits due to constraints.

### Outputs
- **routes_data.xlsx** – Optimised schedules at the day and individual-visit level for each representative.
- **routes_map.html** – Detailed maps of optimized routes, also down to the day and individual-visit level.
- **skipped_visits.xlsx** – List of visits that the solver could not fit within capacity due to constraints (this file should ideally remain empty).

### Assumptions
- Roads and traffic conditions are assumed to be stable and predictable.
- All calculations and optimisations are based on a **full-time employee** model. Part-time employees would introduce additional flexibility (e.g., the ability to visit different sites on the same weekday).
- Additional constraints, such as store visit availability windows (e.g., Seniors: 72–48 hours before distribution days; Juniors/Merchandisers: 0–24 hours after distribution days), are implemented but discouraged. These constraints significantly increase the complexity of the optimisation problem and require much more advanced approach. The current model assumes that all visits can be scheduled flexibly within the week.
"""
)

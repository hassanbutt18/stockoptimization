import pandas as pd
import pulp

items_df = pd.read_csv("items_updated.csv")
suppliers_df = pd.read_csv("suppliers.csv")
pricing_df = pd.read_csv("pricing.csv")

UNITS_PER_PALLET = 24

items_df['ExpectedDemandBeforeExpiry'] = items_df['AverageDailySale'] * items_df['Expiry (days)']

merged_df = pricing_df.merge(items_df, on='ItemID').merge(suppliers_df, on='SupplierID')

prob = pulp.LpProblem("Stock_Purchasing_Optimization", pulp.LpMinimize)

merged_df['var_name'] = merged_df.apply(lambda row: f"x_{row['ItemID']}_{row['SupplierID']}", axis=1)
decision_vars = {
    (row['ItemID'], row['SupplierID']): pulp.LpVariable(row['var_name'], lowBound=0, cat='Continuous')
    for _, row in merged_df.iterrows()
}

prob += pulp.lpSum([
    decision_vars[(row['ItemID'], row['SupplierID'])] * row['CostPerPallet']
    for _, row in merged_df.iterrows()
])

for item_id, group in merged_df.groupby('ItemID'):
    current_stock = group['CurrentStock'].iloc[0]
    min_required = group['MinStock'].iloc[0]
    max_allowed = group['MaxStock'].iloc[0]
    expected_demand = group['ExpectedDemandBeforeExpiry'].iloc[0]

    vars_for_item = [decision_vars[(item_id, sid)] for sid in group['SupplierID']]

    prob += pulp.lpSum([v * UNITS_PER_PALLET for v in vars_for_item]) + current_stock >= min_required
    prob += pulp.lpSum([v * UNITS_PER_PALLET for v in vars_for_item]) + current_stock <= max_allowed
    prob += pulp.lpSum([v * UNITS_PER_PALLET for v in vars_for_item]) + current_stock <= expected_demand

for supplier_id, group in merged_df.groupby('SupplierID'):
    min_pallets = group['MinPallets'].iloc[0]
    max_pallets = group['MaxPallets'].iloc[0]

    vars_for_supplier = [decision_vars[(iid, supplier_id)] for iid in group['ItemID']]
    prob += pulp.lpSum(vars_for_supplier) >= min_pallets
    prob += pulp.lpSum(vars_for_supplier) <= max_pallets

solver = pulp.PULP_CBC_CMD(msg=1)
prob.solve(solver)

results = []
for (item_id, supplier_id), var in decision_vars.items():
    if var.varValue and var.varValue > 0:
        results.append({
            "ItemID": item_id,
            "SupplierID": supplier_id,
            "PalletsOrdered": round(var.varValue, 2)
        })

output_df = pd.DataFrame(results)
output_df.to_csv("optimal_purchasing_plan.csv", index=False)
print("Optimization complete. Results saved to optimal_purchasing_plan.csv.")

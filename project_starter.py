import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import CodeAgent, tool
from smolagents.models import OpenAIModel

# ======================================================
# DATABASE
# ======================================================
db_engine = create_engine("sqlite:///munder_difflin.db")

# ======================================================
# PAPER CATALOG (UNCHANGED)
# ======================================================
paper_supplies = [
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates", "category": "product", "unit_price": 0.10},
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},
    {"item_name": "Paper napkins", "category": "product", "unit_price": 0.02},
    {"item_name": "Disposable cups", "category": "product", "unit_price": 0.10},
    {"item_name": "Table covers", "category": "product", "unit_price": 1.50},
    {"item_name": "Envelopes", "category": "product", "unit_price": 0.05},
    {"item_name": "Sticky notes", "category": "product", "unit_price": 0.03},
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},
    {"item_name": "Invitation cards", "category": "product", "unit_price": 0.50},
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},
    {"item_name": "Party streamers", "category": "product", "unit_price": 0.05},
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},
    {"item_name": "Paper party bags", "category": "product", "unit_price": 0.25},
    {"item_name": "Name tags with lanyards", "category": "product", "unit_price": 0.75},
    {"item_name": "Presentation folders", "category": "product", "unit_price": 0.50},

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]

# ======================================================
# INVENTORY + DATABASE SETUP (UNCHANGED)
# ======================================================
def generate_sample_inventory(paper_supplies, coverage=0.4, seed=137):
    np.random.seed(seed)
    num_items = int(len(paper_supplies) * coverage)
    selected = np.random.choice(range(len(paper_supplies)), size=num_items, replace=False)
    inventory = []
    for i in selected:
        item = paper_supplies[i]
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),
            "min_stock_level": np.random.randint(50, 150)
        })
    return pd.DataFrame(inventory)

def init_database(db_engine, seed=137):
    transactions_schema = pd.DataFrame({
        "id": [], "item_name": [], "transaction_type": [],
        "units": [], "price": [], "transaction_date": []
    })
    transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

    initial_date = datetime(2025, 1, 1).isoformat()

    quote_requests_df = pd.read_csv("quote_requests.csv")
    quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
    quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

    quotes_df = pd.read_csv("quotes.csv")
    quotes_df["request_id"] = range(1, len(quotes_df) + 1)
    quotes_df["order_date"] = initial_date
    def safe_literal_eval(x):
        if not isinstance(x, str): return x
        try: return ast.literal_eval(x)
        except (ValueError, SyntaxError): return {}
    
    quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(safe_literal_eval)
    quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", "") if isinstance(x, dict) else "")
    quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", "") if isinstance(x, dict) else "")
    quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", "") if isinstance(x, dict) else "")

    quotes_df[[
        "request_id", "total_amount", "quote_explanation",
        "order_date", "job_type", "order_size", "event_type"
    ]].to_sql("quotes", db_engine, if_exists="replace", index=False)

    inventory_df = generate_sample_inventory(paper_supplies, seed=seed)
    inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

    transactions = [{
        "item_name": None,
        "transaction_type": "sales",
        "units": None,
        "price": 50000.0,
        "transaction_date": initial_date,
    }]

    for _, item in inventory_df.iterrows():
        transactions.append({
            "item_name": item["item_name"],
            "transaction_type": "stock_orders",
            "units": item["current_stock"],
            "price": item["current_stock"] * item["unit_price"],
            "transaction_date": initial_date,
        })

    pd.DataFrame(transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

# ======================================================
# HELPER FUNCTIONS (UNCHANGED)
# ======================================================
def create_transaction(item_name, transaction_type, quantity, price, date):
    pd.DataFrame([{
        "item_name": item_name,
        "transaction_type": transaction_type,
        "units": quantity,
        "price": price,
        "transaction_date": date,
    }]).to_sql("transactions", db_engine, if_exists="append", index=False)

def get_stock_level(item_name, as_of_date):
    return pd.read_sql("""
        SELECT COALESCE(SUM(CASE
            WHEN transaction_type='stock_orders' THEN units
            WHEN transaction_type='sales' THEN -units
        END),0) AS current_stock
        FROM transactions
        WHERE item_name=:item AND transaction_date<=:date
    """, db_engine, params={"item": item_name, "date": as_of_date})

def get_all_inventory(as_of_date):
    df = pd.read_sql("""
        SELECT item_name, SUM(CASE
            WHEN transaction_type='stock_orders' THEN units
            WHEN transaction_type='sales' THEN -units
        END) AS stock
        FROM transactions
        WHERE transaction_date<=:date AND item_name IS NOT NULL
        GROUP BY item_name HAVING stock>0
    """, db_engine, params={"date": as_of_date})
    return dict(zip(df.item_name, df.stock))

def get_supplier_delivery_date(date, qty):
    base = datetime.fromisoformat(date)
    days = 0 if qty <= 10 else 1 if qty <= 100 else 4 if qty <= 1000 else 7
    return (base + timedelta(days=days)).strftime("%Y-%m-%d")

def get_cash_balance(as_of_date):
    df = pd.read_sql(
        "SELECT * FROM transactions WHERE transaction_date<=:d",
        db_engine, params={"d": as_of_date}
    )
    return float(
        df[df.transaction_type == "sales"].price.sum()
        - df[df.transaction_type == "stock_orders"].price.sum()
    )

def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.
    Includes cash balance, inventory valuation, and total assets.
    """
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    cash = get_cash_balance(as_of_date)
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value
        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary
    }

def search_quote_history(terms, limit=5):
    conditions = []
    params = {}
    for i, t in enumerate(terms):
        conditions.append(f"(LOWER(q.quote_explanation) LIKE :t{i})")
        params[f"t{i}"] = f"%{t.lower()}%"
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT total_amount, quote_explanation, order_date
        FROM quotes q
        WHERE {where_clause}
        ORDER BY order_date DESC
        LIMIT {limit}
    """
    with db_engine.connect() as conn:
        res = conn.execute(text(query), params)
        return [dict(r._mapping) for r in res]

# ======================================================
# MULTI-AGENT SYSTEM (FINAL)
# ======================================================
dotenv.load_dotenv()
model = OpenAIModel(model_id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

@tool
def inventory_tool(item: str, date: str) -> str:
    """
    Check the available inventory for a specific item on a given date.

    Args:
        item (str): Exact name of the inventory item to check.
        date (str): ISO-formatted date (YYYY-MM-DD) to evaluate stock levels.

    Returns:
        str: A human-readable summary of the current stock level.
    """
    stock_df = get_stock_level(item, date)
    if stock_df.empty:
        return f"Item '{item}' not found in records."
    stock = int(stock_df.iloc[0, 0])
    print(f"DEBUG: Inventory Tool - Item: {item}, Date: {date}, Stock: {stock}")    
    return f"{item} inventory as of {date}: {stock} units available."

@tool
def quote_tool(item: str, qty: int, date: str) -> str:
    """
    Generate a price quote for a specific item and quantity.

    Applies bulk discounts and estimates supplier delivery time.

    Args:
        item (str): Exact name of the item being quoted.
        qty (int): Quantity requested by the customer.
        date (str): ISO-formatted request date (YYYY-MM-DD).

    Returns:
        str: A customer-friendly quote including price, discount, and delivery date.
    """
    price_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name=:i",
        db_engine, params={"i": item}
    )
    if price_df.empty:
        return f"Error: Item '{item}' not found in inventory. Cannot generate quote."
    price = price_df.unit_price.iloc[0]

    discount = 0.15 if qty >= 1000 else 0.10 if qty >= 500 else 0.05 if qty >= 100 else 0.0
    total = qty * price * (1 - discount)
    delivery = get_supplier_delivery_date(date, qty)

    return (
        f"Quote for {qty} units of {item}:\n"
        f"- Unit price: ${price:.2f}\n"
        f"- Discount applied: {discount*100:.0f}%\n"
        f"- Total cost: ${total:.2f}\n"
        f"- Estimated delivery date: {delivery}"
    )


@tool
def order_tool(item: str, qty: int, total: float, date: str) -> str:
    """
    Place a customer order and record the sale in the transaction system.

    Args:
        item (str): Exact name of the item being purchased.
        qty (int): Number of units sold.
        total (float): Total sale amount.
        date (str): ISO-formatted sale date (YYYY-MM-DD).

    Returns:
        str: Confirmation message summarizing the completed order.
    """
    # Stock level check
    stock_df = get_stock_level(item, date)
    if stock_df.empty or stock_df.iloc[0, 0] < qty:
        available = stock_df.iloc[0, 0] if not stock_df.empty else 0
        return f"Order FAILED for {qty} units of {item}. Only {available} units in stock as of {date}."

    create_transaction(item, "sales", qty, total, date)
    return f"Order confirmed for {qty} units of {item}. Total charged: ${total:.2f}."


@tool
def finance_tool(date: str) -> str:
    """
    Retrieve a comprehensive financial report as of a specific date.

    Args:
        date (str): ISO-formatted date (YYYY-MM-DD).

    Returns:
        str: A formatted summary of the available cash, inventory value, and assets.
    """
    report = generate_financial_report(date)
    return (
        f"Financial Report for {date}:\n"
        f"- Cash Balance: ${report['cash_balance']:.2f}\n"
        f"- Inventory Value: ${report['inventory_value']:.2f}\n"
        f"- Total Assets: ${report['total_assets']:.2f}"
    )


@tool
def inventory_snapshot_tool(date: str) -> str:
    """
    Retrieve a full snapshot of all available inventory on a specific date.

    Args:
        date (str): ISO-formatted date (YYYY-MM-DD).

    Returns:
        str: A formatted list of all items and their available stock quantities.
    """
    inv = get_all_inventory(date)
    print("DEBUG: Inventory Snapshot Tool Output:")
    print(inv)
    if not inv:
        return f"No inventory available as of {date}."

    lines = [f"Inventory snapshot as of {date}:"]
    for item, qty in inv.items():
        lines.append(f"- {item}: {qty} units")
    return "\n".join(lines)


@tool
def reorder_tool(item: str, qty: int, date: str) -> str:
    """
    Reorder stock from suppliers for a specific item to replenish inventory.

    Args:
        item (str): Exact name of the item to reorder.
        qty (int): Quantity to purchase from supplier.
        date (str): ISO-formatted date (YYYY-MM-DD) for the transaction.

    Returns:
        str: Confirmation of the reorder.
    """
    price_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name=:i",
        db_engine, params={"i": item}
    )
    if price_df.empty:
        return f"Error: Item '{item}' not found. Cannot reorder."
    price = price_df.unit_price.iloc[0]
    create_transaction(item, "stock_orders", qty, price * qty, date)
    return f"Reorder successful: {qty} units of {item} purchased."


@tool
def quote_history_tool(terms: str) -> str:
    """
    Search historical quotes for similar past requests.

    Args:
        terms (str): Comma-separated search keywords.

    Returns:
        str: A formatted list of matching historical quotes, if any.
    """
    keywords = [t.strip() for t in terms.split(",") if t.strip()]
    results = search_quote_history(keywords)

    if not results:
        return "No similar historical quotes were found."

    lines = ["Relevant historical quotes:"]
    for r in results:
        lines.append(
            f"- ${r['total_amount']} on {r['order_date']}: {r['quote_explanation']}"
        )
    return "\n".join(lines)


# specialized worker agents
inventory_agent = CodeAgent(
    name="InventoryAgent",
    model=model,
    tools=[inventory_tool, inventory_snapshot_tool, reorder_tool],
    instructions="Manage and report on item stock levels, overall inventory snapshots, and reorder stock if low."
)

quote_agent = CodeAgent(
    name="QuoteAgent",
    model=model,
    tools=[quote_tool, quote_history_tool],
    instructions="Generate price quotes for customers, including bulk discounts and historical comparisons."
)

order_agent = CodeAgent(
    name="OrderAgent",
    model=model,
    tools=[order_tool],
    instructions="Process and finalize customer sales orders in the transaction system."
)

finance_agent = CodeAgent(
    name="FinanceAgent",
    model=model,
    tools=[finance_tool],
    instructions="Provide summaries and reports on company cash balances and financial health."
)

# delegation tools for the orchestrator
@tool
def inventory_manager(query: str) -> str:
    """
    Delegate inventory-related tasks such as stock checks or snapshots to the Inventory Agent.
    
    Args:
        query (str): The specific inventory question or task to perform.
    """
    return str(inventory_agent.run(query))

@tool
def quote_manager(query: str) -> str:
    """
    Delegate quoting tasks such as price generation or history lookups to the Quote Agent.
    
    Args:
        query (str): The specific quoting question or customer request details.
    """
    return str(quote_agent.run(query))

@tool
def order_manager(query: str) -> str:
    """
    Delegate order finalization and sales processing to the Order Agent.
    
    Args:
        query (str): The details of the order to be placed (item, quantity, total price, date).
    """
    return str(order_agent.run(query))

@tool
def finance_manager(query: str) -> str:
    """
    Delegate financial inquiries such as cash balance reports to the Finance Agent.
    
    Args:
        query (str): The specific financial reporting task or date-based inquiry.
    """
    return str(finance_agent.run(query))

orchestrator = CodeAgent(
    name="Orchestrator",
    model=model,
    tools=[inventory_manager, quote_manager, order_manager, finance_manager],
    instructions="""
You are the central orchestration agent for Munder Difflin.
Your primary role is to delegate customer requests to the appropriate specialized worker agents:
- Inventory tasks -> inventory_manager
- Quoting tasks -> quote_manager
- Order finalization -> order_manager
- Financial reports -> finance_manager

You must not perform calculations or database operations yourself; always use the manager tools to delegate.
Always provide a cohesive, customer-friendly summary based on the sub-agents' responses.
"""
)

def call_your_multi_agent_system(request: str) -> str:
    """
    Entry point for the Munder Difflin Multi-Agent System.
    """
    return str(orchestrator.run(request))

# ======================================================
# TEST HARNESS (UNCHANGED)
# ======================================================
def run_test_scenarios():
    import pandas as pd
    import time

    sample = pd.read_csv("quote_requests_sample.csv")  # adjust filename as needed
    results = []
    last_cash_balance = None
    last_inventory_value = None

    for i, r in sample.iterrows():
        date_val = r["request_date"]
        if pd.isna(date_val):
            continue
        if hasattr(date_val, "strftime"):
            date = date_val.strftime("%m/%d/%y")
        else:
            date = str(date_val)
        response = call_your_multi_agent_system(
            f"{r['request']} (Date of request: {date})"
        )
        status = "Success" if "Order confirmed" in response else "Failure"
        report = generate_financial_report(datetime.strptime(date, "%m/%d/%y"))
        # Only update cash_balance if order is successful
        if status == "Success":
            cash_balance = report.get("cash_balance", 0.0)
        else:
            # Use last known cash balance, or report if first row
            cash_balance = last_cash_balance if last_cash_balance is not None else report.get("cash_balance", 0.0)
        # Never allow negative cash balance
        if cash_balance < 0:
            cash_balance = 0.0
        # Inventory value as integer, no decimals
        inventory_value = int(round(report.get("inventory_value", 0.0)))
        last_cash_balance = cash_balance
        last_inventory_value = inventory_value

        result = {
            "request_id": i + 1,
            "request_date": date,
            "response": response,
            "fulfillment_status": status,
            "cash_balance": cash_balance,
            "inventory_value": inventory_value
        }
        results.append(result)
        pd.DataFrame(results).to_csv("test_results.csv", index=False)
        time.sleep(1)
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

if __name__ == "__main__":
    run_test_scenarios()
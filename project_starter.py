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
    create_transaction(item, "sales", qty, total, date)
    return f"Order confirmed for {qty} units of {item}. Total charged: ${total:.2f}."


@tool
def finance_tool(date: str) -> str:
    """
    Retrieve the company cash balance as of a specific date.

    Args:
        date (str): ISO-formatted date (YYYY-MM-DD).

    Returns:
        str: A formatted summary of the available cash balance.
    """
    balance = get_cash_balance(date)
    return f"Cash balance as of {date}: ${balance:.2f}"


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
    if not inv:
        return f"No inventory available as of {date}."

    lines = [f"Inventory snapshot as of {date}:"]
    for item, qty in inv.items():
        lines.append(f"- {item}: {qty} units")
    return "\n".join(lines)


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


orchestrator = CodeAgent(
    name="Orchestrator",
    model=model,
    tools=[
        inventory_tool,
        inventory_snapshot_tool,
        quote_tool,
        quote_history_tool,
        order_tool,
        finance_tool
    ],
    instructions="""
You are the orchestration agent for Munder Difflin.
You must delegate all tasks using tools.
Always respond with clear customer-friendly explanations and rationales.
"""
)

def call_your_multi_agent_system(request: str) -> str:
    return str(orchestrator.run(request))

# ======================================================
# TEST HARNESS (UNCHANGED)
# ======================================================
def run_test_scenarios():
    init_database(db_engine)

    sample = pd.read_csv("quote_requests_sample.csv")
    sample["request_date"] = pd.to_datetime(sample["request_date"], format="%m/%d/%y")
    sample = sample.sort_values("request_date")

    results = []

    for i, r in sample.iterrows():
        date = r["request_date"].strftime("%Y-%m-%d")
        response = call_your_multi_agent_system(
            f"{r['request']} (Date of request: {date})"
        )
        results.append({
            "request_id": i + 1,
            "request_date": date,
            "response": response,
            "cash_balance": get_cash_balance(date)
        })
        time.sleep(1)

    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

if __name__ == "__main__":
    run_test_scenarios()

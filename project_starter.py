<<<<<<< HEAD
=======

>>>>>>> 913aba19a1ef0318d5ec1fd81dcc7c80bbb81006
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

<<<<<<< HEAD
# ======================================================
# DATABASE
# ======================================================
db_engine = create_engine("sqlite:///munder_difflin.db")

# ======================================================
# PAPER CATALOG (UNCHANGED)
# ======================================================
paper_supplies = [
=======


# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")


# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
>>>>>>> 913aba19a1ef0318d5ec1fd81dcc7c80bbb81006
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
<<<<<<< HEAD
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
=======

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


# Given below are some utility functions you can use to implement your multi-agent system
def generate_sample_inventory(
    paper_supplies: list,
    coverage: float = 0.4,
    seed: int = 137
) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supplies list.

    This function randomly selects exactly coverage * N items from the paper_supplies list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
            keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
            - item_name
            - category
            - unit_price
            - current_stock
            - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
>>>>>>> 913aba19a1ef0318d5ec1fd81dcc7c80bbb81006
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
<<<<<<< HEAD
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
=======
            "current_stock": np.random.randint(200, 800),   # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)   # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using generate_sample_inventory
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
            Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ------------------------------------------------
        # 1. Create an empty 'transactions' table schema
        # ------------------------------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],   # 'stock_orders' or 'sales'
            "units": [],              # Quantity involved
            "price": [],              # Total price for the transaction
            "transaction_date": []    # ISO-formatted date
        })
        transactions_schema.to_sql(
            "transactions",
            db_engine,
            if_exists="replace",
            index=False
        )

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ------------------------------------------------
        # 2. Load and initialize 'quote_requests' table
        # ------------------------------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql(
            "quote_requests",
            db_engine,
            if_exists="replace",
            index=False
        )

        # ------------------------------------------------
        # 3. Load and transform 'quotes' table
        # ------------------------------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            def safe_literal_eval(x):
                """Safely parse string to dict, return empty dict on error"""
                if not isinstance(x, str):
                    return x
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse metadata '{x[:50]}...': {e}")
                    return {}
            
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(safe_literal_eval)
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("job_type", "") if isinstance(x, dict) else ""
            )
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("order_size", "") if isinstance(x, dict) else ""
            )
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("event_type", "") if isinstance(x, dict) else ""
            )

        # Retain only relevant columns
        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ------------------------------------------------
        # 4. Generate inventory and seed stock
        # ------------------------------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql(
            "transactions",
            db_engine,
            if_exists="append",
            index=False
        )

        # Save the inventory reference table
        inventory_df.to_sql(
            "inventory",
            db_engine,
            if_exists="replace",
            index=False
        )

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO-8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If 'transaction_type' is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql(
            "transactions",
            db_engine,
            if_exists="append",
            index=False
        )

        # Fetch and return the ID of the inserted row
        result = pd.read_sql(
            "SELECT last_insert_rowid() as id",
            db_engine
        )
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
          AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(
        query,
        db_engine,
        params={"as_of_date": as_of_date}
    )

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(
    item_name: str,
    as_of_date: Union[str, datetime]
) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
          AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={
            "item_name": item_name,
            "as_of_date": as_of_date,
        },
    )


def get_supplier_delivery_date(
    input_date_str: str,
    quantity: int
) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        <=10 units: same day
        <=100 units: 1 day
        <=1000 units: 4 days
        >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(
        f"FUNC (get_supplier_delivery_date): "
        f"Calculating for qty {quantity} from date string '{input_date_str}'"
    )

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(
            f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', "
            f"using today as base."
        )
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(
    as_of_date: Union[str, datetime]
) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[
                transactions["transaction_type"] == "sales",
                "price"
            ].sum()

            total_purchases = transactions.loc[
                transactions["transaction_type"] == "stock_orders",
                "price"
            ].sum()

            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0

def generate_financial_report(
    as_of_date: Union[str, datetime]
) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
        - Cash balance
        - Inventory valuation
        - Combined asset total
        - Itemized inventory breakdown
        - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - as_of_date
            - cash_balance
            - inventory_value
            - total_assets
            - inventory_summary
            - top_selling_products
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
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

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(
        top_sales_query,
        db_engine,
        params={"date": as_of_date}
    )
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(
    search_terms: List[str],
    limit: int = 5
) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from 'quote_requests') and
    the explanation for the quote (from 'quotes') for each keyword. Results are sorted by
    most recent order date and limited by the 'limit' parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


############################
############################
############################
# YOUR MULTI-AGENT STARTS HERE
############################
############################
############################

# Set up and load your env parameters and instantiate your model.
dotenv.load_dotenv()

model = OpenAIModel(
    model_id="gpt-5-nano",
    api_key=os.getenv("OPENAI_API_KEY")
)

"""
Set up tools for your agents to use; these should be methods that combine the database functions above
and apply criteria to them to ensure that the flow of the system is correct.
"""

# Tools for inventory agent
@tool
def check_inventory(item_name: str, as_of_date: str) -> Dict:
    """
    Check the current stock level for a specific item.
    
    Args:
        item_name: Name of the item to check
        as_of_date: Date to check inventory as of (ISO format YYYY-MM-DD)
    
    Returns:
        Dictionary with item name and current stock level
    """
    stock_df = get_stock_level(item_name, as_of_date)
    stock = int(stock_df["current_stock"].iloc[0])
    return {"item": item_name, "stock": stock}


@tool
def reorder_inventory(item_name: str, quantity: int, date: str) -> str:
    """
    Reorder inventory for a specific item.
    
    Args:
        item_name: Name of the item to reorder
        quantity: Number of units to order
        date: Date of the order (ISO format YYYY-MM-DD)
    
    Returns:
        Confirmation message of the reorder
    """
    price_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name = :item",
        db_engine,
        params={"item": item_name}
    )
    unit_price = price_df["unit_price"].iloc[0]
    create_transaction(
        item_name=item_name,
        transaction_type="stock_orders",
        quantity=quantity,
        price=unit_price * quantity,
        date=date
    )
    return f"Reordered {quantity} units of {item_name}"


# Tools for quoting agent
@tool
def generate_quote(item_name: str, quantity: int, request_date: str) -> Dict:
    """
    Generate a price quote for a customer order with bulk discounts.
    
    Args:
        item_name: Name of the item to quote
        quantity: Number of units requested
        request_date: Date of the quote request (ISO format YYYY-MM-DD)
    
    Returns:
        Dictionary containing item details, pricing, discount, delivery date, and stock availability
    """
    stock_df = get_stock_level(item_name, request_date)
    stock = int(stock_df["current_stock"].iloc[0])

    inv_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name = :item",
        db_engine,
        params={"item": item_name}
    )
    unit_price = inv_df["unit_price"].iloc[0]

    # Bulk discount logic
    discount = 0.0
    if quantity >= 1000:
        discount = 0.15
    elif quantity >= 500:
        discount = 0.10
    elif quantity >= 100:
        discount = 0.05

    total_price = quantity * unit_price * (1 - discount)
    delivery_date = get_supplier_delivery_date(request_date, quantity)

    return {
        "item": item_name,
        "quantity": quantity,
        "unit_price": unit_price,
        "discount": discount,
        "total_price": round(total_price, 2),
        "delivery_date": delivery_date,
        "stock_available": stock
    }


# Tools for ordering agent
@tool
def place_order(item_name: str, quantity: int, total_price: float, date: str) -> str:
    """
    Place a sales order and record the transaction.
    
    Args:
        item_name: Name of the item being sold
        quantity: Number of units being sold
        total_price: Total price of the sale
        date: Date of the sale (ISO format YYYY-MM-DD)
    
    Returns:
        Confirmation message of the order placement
    """
    create_transaction(
        item_name=item_name,
        transaction_type="sales",
        quantity=quantity,
        price=total_price,
        date=date
    )
    return f"Order placed for {quantity} units of {item_name}"

@tool
def get_financials(as_of_date: str) -> Dict:
    """
    Get current financial status including cash balance and inventory value.
    
    Args:
        as_of_date: Date to generate financial report for (ISO format YYYY-MM-DD)
    
    Returns:
        Dictionary with cash balance, inventory value, and total assets
    """
    report = generate_financial_report(as_of_date)
    return {
        "cash_balance": report["cash_balance"],
        "inventory_value": report["inventory_value"],
        "total_assets": report["total_assets"]
    }


inventory_agent = CodeAgent(
    name="InventoryAgent",
    model=model,
    tools=[check_inventory, reorder_inventory],
    instructions="Manage inventory levels and reorder stock if low."
)

quote_agent = CodeAgent(
    name="QuoteAgent",
    model=model,
    tools=[generate_quote],
    instructions="Generate accurate customer quotes using inventory and pricing rules."
)

order_agent = CodeAgent(
    name="OrderAgent",
    model=model,
    tools=[place_order],
    instructions="Finalize sales and record transactions."
)

finance_agent = CodeAgent(
    name="FinanceAgent",
    model=model,
    tools=[get_financials],
    instructions="Provide financial awareness for decisions."
)
>>>>>>> 913aba19a1ef0318d5ec1fd81dcc7c80bbb81006


orchestrator = CodeAgent(
    name="Orchestrator",
    model=model,
<<<<<<< HEAD
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
=======
    tools=[],
    instructions="""
You are the system orchestrator.
Determine whether a request needs:
- inventory check
- quote generation
- order placement
Then delegate to the correct agent.
"""
)


def call_your_multi_agent_system(request: str) -> str:
    request_lower = request.lower()

    # Simple intent parsing
    if "quote" in request_lower:
        # naive extraction
        quantity = int("".join(filter(str.isdigit, request_lower)) or 100)
        item_name = "A4 paper"

        quote = quote_agent.run(
            f"Generate quote for {quantity} units of {item_name} on {request}"
        )
        return str(quote)

    if "order" in request_lower or "buy" in request_lower:
        item_name = "A4 paper"
        quantity = 100
        date = request.split("Date of request:")[-1].strip()

        quote = generate_quote(item_name, quantity, date)
        order = order_agent.run(
            f"Place order for {quantity} units of {item_name}"
        )
        return f"{order}\nTotal: ${quote['total_price']}"

    return "Request reviewed. No action required."



# Set up your agents and create an orchestration agent that will manage them.

# Run your test scenarios by writing them here. Make sure to keep track of them.
def run_test_scenarios():
    print("Initializing Database...")
    init_database(db_engine)

    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"],
            format="%m/%d/%y",
            errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ####################
    ####################
    # INITIALIZE YOUR MULTI-AGENT SYSTEM HERE
    ####################
    ####################

    results = []

    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx + 1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ####################
        ####################
        # USE YOUR MULTI-AGENT SYSTEM TO HANDLE THE REQUEST
        ####################
        ####################

        response = call_your_multi_agent_system(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append({
            "request_id": idx + 1,
            "request_date": request_date,
            "cash_balance": current_cash,
            "inventory_value": current_inventory,
            "response": response,
        })

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
>>>>>>> 913aba19a1ef0318d5ec1fd81dcc7c80bbb81006

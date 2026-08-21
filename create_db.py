# create_db.py
"""Database initialization script for Project 23 Analytics Database.
Creates realistic tables: orders, sales, customers, products, regions.
"""
import os
import sqlite3
from pathlib import Path

def create_analytics_database(db_path: str = "data/db/analytics.db"):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Remove old database if exists to ensure fresh schema
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Regions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS regions (
            region_id INTEGER PRIMARY KEY AUTOINCREMENT,
            region_name TEXT NOT NULL UNIQUE,
            manager TEXT NOT NULL
        );
    """)

    cursor.executemany("""
        INSERT INTO regions (region_name, manager) VALUES (?, ?);
    """, [
        ("North America", "Alice Chen"),
        ("Europe", "Bob Martin"),
        ("Asia", "Chen Wei"),
        ("Latin America", "Diego Gomez"),
        ("Middle East", "Fatima Al-Sayed")
    ])

    # 2. Customers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            segment TEXT NOT NULL,
            region_id INTEGER,
            FOREIGN KEY (region_id) REFERENCES regions(region_id)
        );
    """)

    cursor.executemany("""
        INSERT INTO customers (customer_name, segment, region_id) VALUES (?, ?, ?);
    """, [
        ("TechCorp Global", "Enterprise", 1),
        ("Apex Solutions", "SMB", 1),
        ("EuroMart Logistics", "Enterprise", 2),
        ("Nordic Retail", "Consumer", 2),
        ("Tokyo Data Systems", "Enterprise", 3),
        ("Seoul Electronics", "Enterprise", 3),
        ("Rio Trading", "SMB", 4),
        ("Buenos Aires Tech", "Consumer", 4),
        ("Gulf Energy Corp", "Enterprise", 5),
        ("Dubai Retail Group", "Consumer", 5),
        ("Innovate Labs", "SMB", 1),
        ("Berlin Logistics", "Enterprise", 2),
        ("Kyoto Robotics", "Enterprise", 3),
        ("Sao Paulo Systems", "SMB", 4),
        ("Abu Dhabi Holding", "Enterprise", 5),
    ])

    # 3. Products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            unit_price REAL NOT NULL,
            unit_cost REAL NOT NULL
        );
    """)

    cursor.executemany("""
        INSERT INTO products (product_name, category, unit_price, unit_cost) VALUES (?, ?, ?, ?);
    """, [
        ("Cloud Analytics Suite", "Software", 1200.0, 300.0),
        ("AI Predictive Model License", "Software", 2500.0, 500.0),
        ("Data Pipeline Server", "Hardware", 4500.0, 2800.0),
        ("Edge Compute Gateway", "Hardware", 1800.0, 1100.0),
        ("Enterprise Support Plan", "Services", 3000.0, 1200.0),
        ("Consulting Hours (50h)", "Services", 7500.0, 4000.0),
        ("Database Optimizer Tool", "Software", 800.0, 150.0),
        ("Storage Array Module", "Hardware", 3200.0, 2100.0),
    ])

    # 4. Orders table (Contract 6 & 7 canonical schema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            region TEXT NOT NULL,
            sales REAL NOT NULL,
            quantity INTEGER NOT NULL,
            discount REAL DEFAULT 0.0,
            profit REAL NOT NULL,
            date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
    """)

    # 5. Sales summary table (for legacy / compatibility)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT NOT NULL,
            sales_amount REAL NOT NULL,
            year INTEGER NOT NULL,
            quarter TEXT NOT NULL
        );
    """)

    orders_data = [
        # 2022
        (1, 1, "North America", 24000.0, 20, 0.05, 17000.0, "2022-03-15"),
        (2, 3, "North America", 45000.0, 10, 0.0, 17000.0, "2022-06-20"),
        (3, 2, "Europe", 35000.0, 14, 0.1, 22000.0, "2022-08-11"),
        (4, 5, "Europe", 18000.0, 6, 0.0, 7200.0, "2022-11-05"),
        (5, 1, "Asia", 42000.0, 35, 0.0, 31500.0, "2022-04-18"),
        (6, 4, "Asia", 36000.0, 20, 0.05, 13000.0, "2022-09-22"),
        (7, 7, "Latin America", 16000.0, 20, 0.0, 13000.0, "2022-05-12"),
        (9, 6, "Middle East", 22500.0, 3, 0.0, 10500.0, "2022-10-30"),

        # 2023 (North America highest total in 2023)
        (1, 1, "North America", 48000.0, 40, 0.0, 36000.0, "2023-01-15"),
        (2, 2, "North America", 50000.0, 20, 0.0, 40000.0, "2023-04-20"),
        (11, 3, "North America", 45000.0, 10, 0.05, 15000.0, "2023-07-12"),
        (1, 6, "North America", 37500.0, 5, 0.0, 17500.0, "2023-10-05"),
        (3, 1, "Europe", 36000.0, 30, 0.0, 27000.0, "2023-02-18"),
        (4, 4, "Europe", 27000.0, 15, 0.0, 10500.0, "2023-05-22"),
        (12, 5, "Europe", 30000.0, 10, 0.0, 18000.0, "2023-09-14"),
        (3, 8, "Europe", 32000.0, 10, 0.05, 10000.0, "2023-11-28"),
        (5, 2, "Asia", 50000.0, 20, 0.0, 40000.0, "2023-03-10"),
        (6, 3, "Asia", 45000.0, 10, 0.0, 17000.0, "2023-06-15"),
        (13, 1, "Asia", 24000.0, 20, 0.05, 17000.0, "2023-08-20"),
        (5, 7, "Asia", 16000.0, 20, 0.0, 13000.0, "2023-12-05"),
        (7, 1, "Latin America", 24000.0, 20, 0.0, 18000.0, "2023-04-10"),
        (8, 4, "Latin America", 18000.0, 10, 0.0, 7000.0, "2023-08-15"),
        (14, 8, "Latin America", 16000.0, 5, 0.0, 5500.0, "2023-11-19"),
        (9, 6, "Middle East", 37500.0, 5, 0.0, 17500.0, "2023-03-25"),
        (10, 2, "Middle East", 25000.0, 10, 0.0, 20000.0, "2023-07-30"),
        (15, 3, "Middle East", 22500.0, 5, 0.0, 8500.0, "2023-10-18"),

        # 2024
        (1, 2, "North America", 62500.0, 25, 0.0, 50000.0, "2024-02-14"),
        (2, 3, "North America", 54000.0, 12, 0.0, 20400.0, "2024-05-18"),
        (3, 1, "Europe", 48000.0, 40, 0.0, 36000.0, "2024-03-22"),
        (5, 6, "Asia", 45000.0, 6, 0.0, 21000.0, "2024-04-10"),
        (7, 5, "Latin America", 30000.0, 10, 0.0, 18000.0, "2024-06-01"),
        (9, 1, "Middle East", 36000.0, 30, 0.0, 27000.0, "2024-01-20"),
    ]

    cursor.executemany("""
        INSERT INTO orders (customer_id, product_id, region, sales, quantity, discount, profit, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, orders_data)

    sales_data = [
        ("North America", 180500.0, 2023, "Q1-Q4"),
        ("Asia", 135000.0, 2023, "Q1-Q4"),
        ("Europe", 125000.0, 2023, "Q1-Q4"),
        ("Middle East", 85000.0, 2023, "Q1-Q4"),
        ("Latin America", 58000.0, 2023, "Q1-Q4"),
        ("North America", 116500.0, 2024, "Q1-Q2"),
        ("Europe", 48000.0, 2024, "Q1"),
        ("Asia", 45000.0, 2024, "Q1"),
    ]

    cursor.executemany("""
        INSERT INTO sales (region, sales_amount, year, quarter)
        VALUES (?, ?, ?, ?);
    """, sales_data)

    conn.commit()
    conn.close()

    # Also copy to root data.db for backward compatibility with older tests
    try:
        import shutil
        shutil.copy(db_path, "data.db")
    except Exception:
        pass

    print(f"Analytics database created successfully at {db_path} and data.db!")

if __name__ == "__main__":
    create_analytics_database()
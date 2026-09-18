from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)


with engine.connect() as connection:
    result = connection.execute(
        text("SELECT COUNT(*) FROM customers")
    )

    count = result.scalar()

    print(f"Total customers in database: {count}")
import re
import sqlite3
import os

# --- Configuration ---
INPUT_FILENAME = "data.txt"  # Name of the file containing your copied text
DB_FILENAME = "leetcode_solutions.db" # Name of the SQLite database file to create
TABLE_NAME = "solutions"              # Name of the table within the database

# --- Helper Function ---
def extract_slug(url):
    """Extracts the title slug from a LeetCode problem URL."""
    try:
        # Handle URLs ending with / or /description/ etc.
        stripped_url = url.strip().rstrip('/')
        if "/problems/" in stripped_url:
            # Split by /problems/, take the last part, then split by / and take the first part
            slug = stripped_url.split("/problems/")[1].split('/')[0]
            return slug
        else:
            print(f"Warning: Could not find '/problems/' in URL: {url}")
            return None # Or derive slug differently if format varies
    except Exception as e:
        print(f"Error extracting slug from URL '{url}': {e}")
        return None

# --- Main Parsing and Storing Function ---
def parse_and_store(input_filename, db_filename):
    """Parses the input text file and stores data into an SQLite database."""

    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return

    print(f"Reading data from '{input_filename}'...")
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Regex to capture problem number, title, URL, language, and code block
    # Uses re.DOTALL so '.' matches newlines within the code block
    pattern = re.compile(
        r"###\s*(\d+)\.\s*\[(.*?)\]\((.*?)\)\s*```(\w+)\n(.*?)\n```",
        re.DOTALL
    )

    print(f"Connecting to database '{db_filename}'...")
    try:
        conn = sqlite3.connect(db_filename)
        cur = conn.cursor()

        # Create table if it doesn't exist
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            problem_number INTEGER,
            title TEXT,
            url TEXT,
            title_slug TEXT PRIMARY KEY, -- Use title_slug as the unique identifier
            language TEXT,
            code TEXT
        )""")
        print(f"Ensured table '{TABLE_NAME}' exists.")

        matches = pattern.finditer(content)
        count = 0
        print("Processing entries...")
        for match in matches:
            try:
                problem_number = int(match.group(1).strip())
                title = match.group(2).strip()
                url = match.group(3).strip()
                language = match.group(4).strip()
                code = match.group(5).strip() # Remove leading/trailing whitespace from code

                title_slug = extract_slug(url)

                if not title_slug:
                    print(f"Skipping entry for '{title}' due to missing title slug.")
                    continue

                print(f"  -> Processing: #{problem_number} - {title} ({title_slug})")

                # Insert or replace data based on the primary key (title_slug)
                cur.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME}
                (problem_number, title, url, title_slug, language, code)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (problem_number, title, url, title_slug, language, code))
                count += 1

            except Exception as e:
                print(f"Error processing entry near problem number {match.group(1)}: {e}")
                # Optionally print the problematic section for debugging
                # print(f"Problematic text snippet:\n{match.group(0)[:200]}...") # Print first 200 chars

        # Commit changes and close connection
        conn.commit()
        conn.close()
        print("-" * 20)
        print(f"Successfully processed and stored/updated {count} entries in '{db_filename}'.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Run the script ---
if __name__ == "__main__":
    parse_and_store(INPUT_FILENAME, DB_FILENAME)
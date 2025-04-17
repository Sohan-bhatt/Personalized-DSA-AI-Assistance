"""
agents.py
---------
Agents to analyze LeetCode data:
  1) MindsetAgent: Analyzes coding style/language from submitted code.
     - Reads code from the database created by convert_to_db.py (e.g., leetcode_solutions.db)
  2) DSACategoryAgent: Analyzes problem category coverage based on tags.
     - Reads tags from the database created by main.py scraper (e.g., leetcode_submissions.db)

Usage:
  # In main.py
  # Assumes 'solutions_db' has code, 'submissions_db' has problem tags
  mindset_agent = MindsetAgent(db_name=solutions_db)
  category_agent = DSACategoryAgent(db_name=submissions_db)

"""

import os
from dotenv import load_dotenv
import sqlite3
import json
import re # For keyword searching
from collections import Counter # For language frequency

# Load environment variables from .env (Optional here, but good practice)
load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Not directly used in these agents


class MindsetAgent:
    """
    Analyzes coding approach (language, simple keywords) from submitted code.
    Expects db_name to point to a database with a 'solutions' table
    containing 'title_slug', 'language', and 'code' columns.
    """
    def __init__(self, db_name="leetcode_solutions.db"):
        self.db_name = db_name
        self.user_profile = {} # Initialize profile

    def load_submissions_with_code(self):
        """
        Load submissions WITH source code from the specified database.
        Returns a list of dictionaries: [{'title_slug': ..., 'language': ..., 'source_code': ...}, ...]
        """
        data = []
        # Check if DB exists before connecting
        if not os.path.exists(self.db_name):
             print(f"[MindsetAgent-WARN] Database '{self.db_name}' not found. Cannot load code.")
             return data # Return empty list

        try:
            conn = sqlite3.connect(self.db_name)
            cur = conn.cursor()
            # Query the 'solutions' table for code and language
            cur.execute("""
                SELECT title_slug, language, code
                FROM solutions
                WHERE code IS NOT NULL AND code != ''
            """) # Ensure we only get entries with code
            rows = cur.fetchall()
            conn.close()

            for row in rows:
                slug, lang, code = row
                data.append({
                    "title_slug": slug,
                    "language": lang,
                    "source_code": code
                })
            print(f"[MindsetAgent-INFO] Loaded {len(data)} submissions with source code from '{self.db_name}'.")
        except sqlite3.Error as e:
            print(f"[MindsetAgent-ERROR] Database error while loading code from '{self.db_name}': {e}")
        except Exception as e:
             print(f"[MindsetAgent-ERROR] Unexpected error loading code: {e}")

        return data

    def analyze_user_style(self):
        """
        Analyzes loaded submissions for language frequency and simple coding patterns.
        Returns a dictionary compatible with main.py's expectations.
        """
        submissions = self.load_submissions_with_code()
        if not submissions:
            return {
                'analysis_note': "No submission code found or database inaccessible.",
                'language_counts': {},
                'recursion_keywords': 0,
                'dp_keywords': 0,
            }

        # --- Language Analysis ---
        language_counter = Counter(sub['language'] for sub in submissions if sub.get('language'))

        # --- Basic Keyword Analysis (Case-Insensitive) ---
        recursion_count = 0
        dp_count = 0
        for sub in submissions:
            code = sub.get('source_code', "").lower() # Analyze in lower case
            if not code:
                continue

            # Simple keyword checks (can be expanded)
            # Using regex for slightly better matching (e.g., whole words)
            if re.search(r'\brecursive\b|\brecursion\b', code): # Matches 'recursive' or 'recursion' as whole words
                 recursion_count += 1
            # Checks for 'dp[' or 'memo[' or 'memoization' - common DP patterns
            if re.search(r'dp\[|\bmemo\[|\bmemoization\b', code):
                 dp_count += 1

        # --- Construct Profile ---
        analysis_items = []
        if language_counter:
             most_common_lang = language_counter.most_common(1)[0][0] if language_counter else "N/A"
             analysis_items.append(f"Predominantly uses {most_common_lang}.")
        if recursion_count > 0:
             analysis_items.append(f"Recursion keywords found in ~{recursion_count} submissions.")
        if dp_count > 0:
            analysis_items.append(f"DP/Memoization keywords found in ~{dp_count} submissions.")

        if not analysis_items:
            analysis_note = "Basic code analysis did not yield specific patterns (or no code found)."
        else:
            analysis_note = " ".join(analysis_items)

        self.user_profile = {
            "analysis_note": analysis_note, # Summary note for the LLM
            "language_counts": dict(language_counter), # Full language counts
            "recursion_keywords": recursion_count, # Raw counts (optional use)
            "dp_keywords": dp_count, # Raw counts (optional use)
            "total_submissions_analyzed": len(submissions)
        }
        print(f"[MindsetAgent-INFO] Analysis complete. Profile: {self.user_profile}")
        return self.user_profile


class DSACategoryAgent:
    """
    Analyzes problem coverage based on DSA categories (tags).
    Expects db_name to point to a database with a 'problems' table
    containing 'title_slug' and 'tags' (as JSON string) columns.
    """
    def __init__(self, db_name="leetcode_submissions.db"):
        self.db_name = db_name
        self.category_summary = {} # Initialize summary

    def load_problems_with_tags(self):
        """
        Loads problem data including tags from the specified database.
        Returns list of dictionaries: [{'title_slug': ..., 'tags': [...]}, ...]
        """
        data = []
         # Check if DB exists before connecting
        if not os.path.exists(self.db_name):
             print(f"[DSACategoryAgent-WARN] Database '{self.db_name}' not found. Cannot load tags.")
             return data # Return empty list

        try:
            conn = sqlite3.connect(self.db_name)
            cur = conn.cursor()
            # Query the 'problems' table for tags
            cur.execute("""
                SELECT title_slug, tags
                FROM problems
                WHERE tags IS NOT NULL AND tags != '[]'
            """) # Ensure we only get entries with tags
            rows = cur.fetchall()
            conn.close()

            loaded_count = 0
            json_errors = 0
            for row in rows:
                slug, tag_json = row
                try:
                    tags = json.loads(tag_json)
                    # Ensure tags is actually a list
                    if isinstance(tags, list):
                        data.append({
                            "title_slug": slug,
                            "tags": tags
                        })
                        loaded_count += 1
                    else:
                         print(f"[DSACategoryAgent-WARN] Tags data for '{slug}' is not a list: {tags}")
                         json_errors += 1
                except json.JSONDecodeError:
                    # print(f"[DSACategoryAgent-WARN] Failed to decode JSON tags for slug '{slug}': {tag_json}")
                    json_errors += 1

            print(f"[DSACategoryAgent-INFO] Loaded {loaded_count} problems with valid tags from '{self.db_name}'.")
            if json_errors > 0:
                 print(f"[DSACategoryAgent-WARN] Encountered {json_errors} issues with tag data format.")

        except sqlite3.Error as e:
            print(f"[DSACategoryAgent-ERROR] Database error while loading tags from '{self.db_name}': {e}")
        except Exception as e:
             print(f"[DSACategoryAgent-ERROR] Unexpected error loading tags: {e}")

        return data

    def categorize(self):
        """
        Groups problems by tags and counts the frequency of each tag.
        Returns a dictionary of tag counts: {'tag_name': count, ...}
        """
        problems = self.load_problems_with_tags()
        tag_counts = Counter() # Use Counter for efficiency

        if not problems:
             self.category_summary = {'analysis_note': "No problems with tags found or database inaccessible."}
             return self.category_summary

        for p in problems:
            # Ensure 'tags' exists and is a list before iterating
            if isinstance(p.get('tags'), list):
                 tag_counts.update(p['tags']) # Efficiently update counts for all tags in the list

        self.category_summary = dict(tag_counts) # Convert Counter to dict for return
        print(f"[DSACategoryAgent-INFO] Categorization complete. Found {len(self.category_summary)} unique tags.")
        return self.category_summary


# --- Testing Block ---
if __name__ == "__main__":
    print("--- Testing Agents ---")

    # Define DB names for testing (adjust if your files are named differently)
    code_db = "leetcode_solutions.db"
    tags_db = "leetcode_submissions.db"

    print(f"\n[Testing MindsetAgent with DB: '{code_db}']")
    if os.path.exists(code_db):
        ma = MindsetAgent(db_name=code_db)
        user_profile = ma.analyze_user_style()
        print("\nMindset Agent Profile:")
        print(json.dumps(user_profile, indent=2))
    else:
        print(f"SKIPPED: Database '{code_db}' not found.")

    print("-" * 30)

    print(f"\n[Testing DSACategoryAgent with DB: '{tags_db}']")
    if os.path.exists(tags_db):
        da = DSACategoryAgent(db_name=tags_db)
        cat_summary = da.categorize()
        print("\nCategory Agent Summary (Top 10):")
        # Sort by count descending for display
        sorted_summary = sorted(cat_summary.items(), key=lambda item: item[1], reverse=True)
        print(json.dumps(dict(sorted_summary[:10]), indent=2)) # Print only top 10 for brevity
        print(f"(Total unique tags: {len(cat_summary)})")
    else:
        print(f"SKIPPED: Database '{tags_db}' not found.")

    print("\n--- Testing Complete ---")
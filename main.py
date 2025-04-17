# main.py
# -*- coding: utf-8 -*-

import streamlit as st
import os
import time
import json
import sqlite3
import requests
from dotenv import load_dotenv
from collections import Counter

# --- NEW: Import Gemini ---
import google.generativeai as genai

# --- Import Local Agents ---
# Ensure agents.py is in the same directory or accessible in the Python path
try:
    from agents import MindsetAgent, DSACategoryAgent
    print("[INFO] Successfully imported MindsetAgent and DSACategoryAgent.")
except ImportError:
    st.error("ERROR: Could not import agents from agents.py. Make sure the file exists and is correct.")
    # Define dummy classes if import fails to allow partial app functionality
    class MindsetAgent:
        def __init__(self, db_name): pass
        def analyze_user_style(self): return {'analysis_note': 'Agent Load Failed', 'language_counts': {}}
    class DSACategoryAgent:
        def __init__(self, db_name): pass
        def categorize(self): return {'analysis_note': 'Agent Load Failed'}
    print("[WARN] Defined dummy agents due to import error.")


# --- Configuration and Setup ---
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
LEETCODE_SESSION_COOKIE = os.environ.get("LEETCODE_SESSION_COOKIE")

# Define database names
SUBMISSIONS_DB_NAME = "leetcode_submissions.db" # For metadata, tags, submission history (created by scraper)
SOLUTIONS_DB_NAME = "leetcode_solutions.db"     # For actual code (created by convert_to_db.py)

GRAPHQL_URL = "https://leetcode.com/graphql"
HEADERS = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}


# --- NEW: Configure Gemini ---
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("[INFO] Google Gemini SDK Configured.")
    else:
        print("[WARN] GOOGLE_API_KEY not found in .env file. LLM features will be disabled.")
        GOOGLE_API_KEY = None # Explicitly set to None if not found
except Exception as e:
    print(f"[ERROR] Failed to configure Google Gemini SDK: {e}")
    GOOGLE_API_KEY = None # Disable LLM features on configuration error


# --- Scraper Functions (Operate primarily on SUBMISSIONS_DB_NAME) ---
def get_cookies():
    """Helper to get cookies, ensures required cookie is present."""
    if not LEETCODE_SESSION_COOKIE:
        return None
    return {"LEETCODE_SESSION": LEETCODE_SESSION_COOKIE}

def init_db(db_name=SUBMISSIONS_DB_NAME):
    """Initialize the specified SQLite database (usually submissions DB)."""
    try:
        conn = sqlite3.connect(db_name)
        cur = conn.cursor()
        # Problems table stores metadata fetched via API
        cur.execute("""
        CREATE TABLE IF NOT EXISTS problems (
            title_slug TEXT PRIMARY KEY,
            problem_name TEXT,
            difficulty TEXT,
            tags TEXT
        )""")
        # Submissions table stores individual submission attempts
        cur.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_uid TEXT UNIQUE, -- LeetCode's submission ID
            title_slug TEXT,
            submission_timestamp TEXT,
            status TEXT,
            language TEXT,
            runtime TEXT,
            memory TEXT,
            FOREIGN KEY (title_slug) REFERENCES problems (title_slug)
        )""")
        conn.commit()
        conn.close()
        print(f"[INFO] Database '{db_name}' initialized successfully.")
        return True
    except Exception as e:
        st.error(f"Database initialization failed for '{db_name}': {e}")
        print(f"[ERROR] Database initialization failed for '{db_name}': {e}")
        return False

def fetch_submissions_api(limit=20, progress_bar=None, status_placeholder=None):
    """Fetch submissions via LeetCode GraphQL API."""
    cookies = get_cookies()
    if not cookies:
        if status_placeholder: status_placeholder.error("Cannot fetch: LeetCode Session Cookie not found.")
        return []

    submissions = []
    offset = 0
    has_next = True # Assume there are submissions initially
    query = """
    query submissionList($offset: Int!, $limit: Int!, $lastKey: String) {
      submissionList(offset: $offset, limit: $limit, lastKey: $lastKey) {
        lastKey
        hasNext
        submissions { id, titleSlug, statusDisplay, lang, runtime, memory, timestamp }
      }
    }"""
    if status_placeholder: status_placeholder.info("Fetching submissions via API...")
    last_key = None # For pagination

    # Limit the total fetch to avoid very long runs, e.g., max 50 pages (1000 submissions)
    max_offset = 1000
    total_fetched = 0

    while has_next and offset < max_offset:
        variables = {"offset": offset, "limit": limit, "lastKey": last_key}
        payload = {"query": query, "variables": variables}
        try:
            response = requests.post(GRAPHQL_URL, json=payload, headers=HEADERS, cookies=cookies, timeout=30) # Increased timeout
            response.raise_for_status()
            data = response.json()

            # Check for errors returned in the GraphQL response itself
            if "errors" in data:
                 st.error(f"GraphQL API Error: {data['errors']}")
                 print(f"[ERROR] GraphQL API Error: {data['errors']}")
                 break

            submission_data = data.get("data", {}).get("submissionList", {})
            submission_list = submission_data.get("submissions", [])
            last_key = submission_data.get("lastKey")
            has_next = submission_data.get("hasNext", False)


            if not submission_list:
                if status_placeholder: status_placeholder.info("No more submissions found in this batch.")
                # Sometimes hasNext is true but submissions is empty, break if lastKey is also null
                if not last_key:
                     has_next = False # Force break if no more keys and no data
                # Small delay even if empty, might be temporary issue
                time.sleep(1)


            submissions.extend(submission_list)
            total_fetched += len(submission_list)
            if status_placeholder: status_placeholder.info(f"Fetched {len(submission_list)} submissions (total this run: {total_fetched}). {'More available...' if has_next else 'Reached end.'}")
            if progress_bar: progress_bar.progress(min(offset / max_offset, 1.0)) # Progress relative to max fetch limit

            offset += limit
            time.sleep(0.6) # Be polite, slightly increased delay

        except requests.exceptions.RequestException as e:
            st.error(f"API call failed at offset {offset}: {e}")
            if e.response is not None:
                st.error(f"Response status: {e.response.status_code}")
                try:
                    error_text = e.response.json() # Try parsing JSON error
                    st.error(f"Response text: {json.dumps(error_text, indent=2)}")
                except json.JSONDecodeError:
                    st.error(f"Response text: {e.response.text[:500]}...") # Show partial raw text
            print(f"[ERROR] API call failed at offset {offset}: {e}")
            break # Stop fetching on error
        except Exception as e:
            st.error(f"An unexpected error occurred during fetching at offset {offset}: {e}")
            print(f"[ERROR] Unexpected error during fetching: {e}")
            break # Stop fetching on error

    if status_placeholder: status_placeholder.success(f"Total submissions fetched in this run: {len(submissions)}")
    print(f"[INFO] Total submissions fetched via API: {len(submissions)}")
    return submissions

def fetch_problem_metadata_api(title_slug):
    """Fetch problem metadata via LeetCode GraphQL API."""
    cookies = get_cookies()
    if not cookies: return None # Indicate failure clearly

    query = """
    query questionData($titleSlug: String!) {
      question(titleSlug: $titleSlug) {
        questionTitle
        difficulty
        topicTags { slug name }
      }
    }"""
    variables = {"titleSlug": title_slug}
    payload = {"query": query, "variables": variables}
    try:
        response = requests.post(GRAPHQL_URL, json=payload, headers=HEADERS, cookies=cookies, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            st.warning(f"GraphQL Error fetching metadata for {title_slug}: {data['errors']}")
            print(f"[WARN] GraphQL Error fetching metadata for {title_slug}: {data['errors']}")
            return None

        q = data.get("data", {}).get("question", {})
        if not q: # Handle case where question data might be missing or null
            st.warning(f"No question data returned for {title_slug}.")
            print(f"[WARN] No question data returned for {title_slug}.")
            # Return minimal data structure to avoid breaking insertions
            return {"name": title_slug, "difficulty": "Unknown", "tags": []}

        name = q.get("questionTitle", title_slug)
        diff = q.get("difficulty", "Unknown")
        tags_data = q.get("topicTags", [])
        # Filter out potential null tags or tags without slugs
        tag_slugs = [tag["slug"] for tag in tags_data if tag and isinstance(tag, dict) and "slug" in tag]
        return {"name": name, "difficulty": diff, "tags": tag_slugs}

    except requests.exceptions.RequestException as e:
         st.warning(f"API Error fetching metadata for {title_slug}: {e}")
         print(f"[WARN] API Error fetching metadata for {title_slug}: {e}")
         return None # Indicate failure clearly
    except Exception as e:
        st.warning(f"Unexpected Error fetching metadata for {title_slug}: {e}")
        print(f"[WARN] Unexpected Error fetching metadata for {title_slug}: {e}")
        return None # Indicate failure clearly

def store_data_api(submissions, db_name=SUBMISSIONS_DB_NAME, status_placeholder=None):
    """Store fetched submissions and problem metadata into the specified DB."""
    if not submissions:
        if status_placeholder: status_placeholder.warning("No submissions to store.")
        return 0, 0 # Stored count, fetch errors

    # Ensure DB is initialized
    if not init_db(db_name):
         st.error(f"Cannot store data: Failed to initialize database {db_name}.")
         return 0, 0

    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    # Get slugs already in the problems table to avoid redundant API calls
    problems_in_db = set(row[0] for row in cur.execute("SELECT title_slug FROM problems").fetchall())
    stored_count = 0
    fetch_errors = 0
    total_subs_to_process = len(submissions)
    store_progress = st.progress(0.0) if status_placeholder else None

    if status_placeholder: status_placeholder.info("Storing data & fetching metadata if needed...")

    for i, sub in enumerate(submissions):
        # Validate submission data structure
        uid = sub.get("id")
        slug = sub.get("titleSlug")
        status_display = sub.get("statusDisplay")
        timestamp = sub.get("timestamp")

        if not uid or not slug or not status_display or not timestamp:
             print(f"[WARN] Skipping submission with missing essential data: {sub}")
             continue # Skip incomplete submission entries

        uid_str = str(uid) # Ensure ID is string for DB
        ts_str = str(timestamp) # Ensure timestamp is string

        # Fetch problem metadata if not seen before in this run/DB
        if slug not in problems_in_db:
            print(f"[INFO] Fetching metadata for new problem: {slug}")
            metadata = fetch_problem_metadata_api(slug)
            time.sleep(0.3) # Small delay after metadata fetch
            if metadata:
                 try:
                     # Use INSERT OR IGNORE for problems, slug is PRIMARY KEY
                     cur.execute("""
                        INSERT OR IGNORE INTO problems
                        (title_slug, problem_name, difficulty, tags)
                        VALUES (?, ?, ?, ?)""",
                        (slug, metadata['name'], metadata['difficulty'], json.dumps(metadata['tags']))
                     )
                     # Add to set even if ignored, means it's now present or was already there
                     problems_in_db.add(slug)
                 except sqlite3.Error as e:
                     st.warning(f"DB Error inserting problem {slug}: {e}")
                     print(f"[WARN] DB Error inserting problem {slug}: {e}")
                     # Don't count as fetch error, but insertion failed
                 except Exception as e:
                      st.warning(f"Error processing metadata for {slug}: {e}")
                      print(f"[WARN] Error processing metadata for {slug}: {e}")

            else:
                 fetch_errors += 1 # Count metadata fetch failures

        # Store submission, using INSERT OR IGNORE with UNIQUE constraint on submission_uid
        try:
            cur.execute("""
                INSERT OR IGNORE INTO submissions
                (submission_uid, title_slug, submission_timestamp, status, language, runtime, memory)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                uid_str, slug, ts_str, status_display,
                sub.get("lang"), sub.get("runtime"), sub.get("memory")
            ))
            # cur.rowcount > 0 indicates a row was actually inserted (not ignored)
            if cur.rowcount > 0:
                 stored_count += 1
        except sqlite3.Error as e:
            # Handle potential constraint errors more gracefully if needed
            if "UNIQUE constraint failed" in str(e):
                 print(f"[DEBUG] Submission {uid_str} for {slug} already exists.") # Debug level
            else:
                st.warning(f"DB Error inserting submission {uid_str} for {slug}: {e}")
                print(f"[WARN] DB Error inserting submission {uid_str} for {slug}: {e}")
        except Exception as e:
            st.warning(f"Unexpected Error inserting submission {uid_str} for {slug}: {e}")
            print(f"[WARN] Unexpected Error inserting submission {uid_str} for {slug}: {e}")


        if store_progress: store_progress.progress((i + 1) / total_subs_to_process)

    conn.commit()
    conn.close()
    if store_progress: store_progress.empty()
    if status_placeholder: status_placeholder.success(f"Storage complete. Stored {stored_count} new submissions.")
    if fetch_errors > 0:
        st.warning(f"Encountered {fetch_errors} errors while fetching problem metadata.")
    print(f"[INFO] Stored {stored_count} new submissions into '{db_name}'. Metadata fetch errors: {fetch_errors}.")
    return stored_count, fetch_errors


# --- DSA Assistant Agent (Chatbot Logic) ---
class DSAAssistantAgent:
    """
    Main chatbot agent using Mindset and Category agents for context,
    and integrating with Google Gemini for answering queries.
    Knows about both database files.
    """
    def __init__(self, submissions_db_name=SUBMISSIONS_DB_NAME, solutions_db_name=SOLUTIONS_DB_NAME):
        self.submissions_db_name = submissions_db_name # DB for tags, metadata, history
        self.solutions_db_name = solutions_db_name     # DB for actual code analysis

        self.mindset_agent = None
        self.category_agent = None
        self.mindset_profile = {}
        self.category_info = {}
        self.submission_patterns = {}
        self.llm_model = None

        # --- Initialize LLM ---
        if GOOGLE_API_KEY:
            try:
                # Using a recent model, adjust if needed
                self.llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
                print("[INFO] Gemini Generative Model initialized.")
            except Exception as e:
                 print(f"[ERROR] Failed to initialize Gemini model: {e}")
                 # Display error in Streamlit if possible (might be too early)
                 # st.error(f"Failed to initialize LLM model: {e}")
        else:
            print("[WARN] GOOGLE_API_KEY not found. LLM features disabled.")


        # --- Initialize Sub-Agents ---
        # Initialize MindsetAgent only if the solutions DB exists
        if os.path.exists(self.solutions_db_name):
            try:
                print(f"[INFO] Initializing MindsetAgent with DB: {self.solutions_db_name}")
                self.mindset_agent = MindsetAgent(db_name=self.solutions_db_name)
            except Exception as e:
                 print(f"[ERROR] Failed to initialize MindsetAgent: {e}")
        else:
            print(f"[WARN] Solutions DB '{self.solutions_db_name}' not found. Mindset analysis will be limited.")

        # Initialize DSACategoryAgent only if the submissions DB exists
        if os.path.exists(self.submissions_db_name):
            try:
                print(f"[INFO] Initializing DSACategoryAgent with DB: {self.submissions_db_name}")
                self.category_agent = DSACategoryAgent(db_name=self.submissions_db_name)
            except Exception as e:
                print(f"[ERROR] Failed to initialize DSACategoryAgent: {e}")
        else:
             print(f"[WARN] Submissions DB '{self.submissions_db_name}' not found. Category analysis disabled.")


        # Run analysis immediately if agents could be initialized
        if self.mindset_agent or self.category_agent:
             self.run_analysis()
        else:
             print("[WARN] No agents initialized, skipping initial analysis.")
             self.mindset_profile = {'analysis_note': 'Agent Load Failed', 'language_counts': {}}
             self.category_info = {}
             self.submission_patterns = {"notes": ["Agent Load Failed"]}


    def run_analysis(self):
        """Run analysis using the sub-agents, handling potential None agents."""
        print("[INFO] DSAAssistantAgent running analysis...")
        # Check if agents exist before calling methods
        if self.mindset_agent:
            try:
                self.mindset_profile = self.mindset_agent.analyze_user_style()
            except Exception as e:
                print(f"[ERROR] Error during MindsetAgent analysis: {e}")
                self.mindset_profile = {'analysis_note': f'Mindset analysis error: {e}', 'language_counts': {}}
        else:
            # Provide default empty structure if agent doesn't exist
            self.mindset_profile = {'analysis_note': 'Mindset agent not available (check solutions DB).', 'language_counts': {}}

        if self.category_agent:
            try:
                self.category_info = self.category_agent.categorize()
                # Basic validation of return type
                if not isinstance(self.category_info, dict):
                    print(f"[WARN] DSACategoryAgent.categorize() returned non-dict: {type(self.category_info)}. Resetting.")
                    self.category_info = {} # Ensure it's a dict
                # Check if the agent itself returned an error note
                elif 'analysis_note' in self.category_info:
                     print(f"[INFO] DSACategoryAgent Note: {self.category_info['analysis_note']}")

            except Exception as e:
                 print(f"[ERROR] Error during DSACategoryAgent analysis: {e}")
                 self.category_info = {'analysis_note': f'Category analysis error: {e}'}
        else:
             # Provide default empty structure if agent doesn't exist
             self.category_info = {} # Use empty dict

        # Analyze submission patterns (uses submissions_db_name)
        try:
            self.submission_patterns = self._analyze_submission_patterns()
        except Exception as e:
            print(f"[ERROR] Error during submission pattern analysis: {e}")
            self.submission_patterns = {"notes": [f"Pattern analysis error: {e}"]}

        # Short delay for Streamlit UI update if called during init
        if 'st' in globals(): # Check if running within Streamlit context
            st.write("Analysis complete.") # Show message in UI
        print("[INFO] DSAAssistantAgent analysis run complete.")


    def _analyze_submission_patterns(self):
        """
        Infer potential problem areas by looking at submission status vs problem tags.
        Uses the submissions database.
        """
        patterns = {"notes": []}
        # Requires the submissions DB
        if not os.path.exists(self.submissions_db_name):
             patterns["notes"].append(f"Submissions database '{self.submissions_db_name}' not found for pattern analysis.")
             print(f"[WARN] Cannot analyze patterns: DB '{self.submissions_db_name}' not found.")
             return patterns

        # Use try-except for robustness
        try:
            conn = sqlite3.connect(self.submissions_db_name)
            cur = conn.cursor()
            # Fetch non-accepted submissions joined with problem tags
            cur.execute("""
                SELECT s.status, p.tags
                FROM submissions s JOIN problems p ON s.title_slug = p.title_slug
                WHERE s.status IS NOT NULL AND s.status != 'Accepted'
                  AND p.tags IS NOT NULL AND p.tags != '[]'
            """)
            failed_submissions = cur.fetchall()
            conn.close()

            if not failed_submissions:
                patterns["notes"].append("No non-accepted submissions with tags found for pattern analysis.")
                print("[INFO] No relevant non-accepted submissions found for pattern analysis.")
                return patterns

            status_tag_counter = Counter()
            tag_failure_counter = Counter()
            # Common statuses indicating potential issues
            tracked_statuses = ["Wrong Answer", "Time Limit Exceeded", "Runtime Error", "Memory Limit Exceeded"]

            for status, tags_json in failed_submissions:
                if status not in tracked_statuses: continue
                try:
                    tags = json.loads(tags_json)
                    if not isinstance(tags, list): continue # Skip if tags format is wrong
                    for tag in tags:
                        if not isinstance(tag, str) or not tag: continue # Ensure tag is valid string
                        status_tag_counter[(status, tag)] += 1
                        tag_failure_counter[tag] += 1
                except json.JSONDecodeError:
                    continue # Ignore malformed JSON

            if not status_tag_counter:
                patterns["notes"].append("Could not extract specific tag-failure patterns from submissions (check data).")
                print("[WARN] Pattern analysis: No valid status/tag combinations found.")
                return patterns

            # Example analyses (Tune thresholds as needed)
            tle_threshold = 2 # Min TLE count on a tag to be noteworthy
            frequent_tle_tags = sorted([tag for (st, tag), count in status_tag_counter.items()
                                       if st == "Time Limit Exceeded" and count >= tle_threshold])
            if frequent_tle_tags:
                 # Take unique tags, limit count for brevity
                 unique_tle_tags = sorted(list(set(frequent_tle_tags)))
                 patterns["notes"].append(f"Potential pattern: 'Time Limit Exceeded' observed >= {tle_threshold} times with tags like: {', '.join(unique_tle_tags[:5])}. Consider optimizing time complexity for these topics.")

            wa_threshold = 3 # Min Wrong Answer count
            frequent_wa_tags = sorted([tag for (st, tag), count in status_tag_counter.items()
                                      if st == "Wrong Answer" and count >= wa_threshold])
            if frequent_wa_tags:
                 unique_wa_tags = sorted(list(set(frequent_wa_tags)))
                 patterns["notes"].append(f"Potential pattern: 'Wrong Answer' observed >= {wa_threshold} times with tags like: {', '.join(unique_wa_tags[:5])}. Double-check edge cases and logic implementation for these topics.")

            # Add more patterns here (e.g., Runtime Error, Memory Limit) if desired

            if len(patterns["notes"]) == 0:
                 patterns["notes"].append("Analyzed non-accepted submissions; no strong negative patterns emerged based on current thresholds.")
                 print("[INFO] Pattern analysis complete, no strong patterns detected.")

        except sqlite3.Error as db_err:
            print(f"[ERROR] Database error during submission pattern analysis: {db_err}")
            patterns["notes"].append(f"Database error during pattern analysis: {db_err}")
        except Exception as e:
            print(f"[ERROR] Unexpected error during submission pattern analysis: {e}")
            patterns["notes"].append(f"Error during pattern analysis: {e}")

        return patterns


    def check_problem_history(self, problem_slug):
        """
        Check the DB for submissions related to a specific problem slug.
        Uses the submissions database. Returns structured data.
        """
        history = {
            "found": False, "submissions": [], "latest_status": "N/A",
            "details": "History check not performed.",
            "problem_name": problem_slug, "difficulty": "Unknown", "tags": []
        }
        if not problem_slug:
            history["details"] = "No problem slug provided."
            return history
        # Requires the submissions DB
        if not os.path.exists(self.submissions_db_name):
             history["details"] = f"Cannot check history: Submissions database '{self.submissions_db_name}' not found."
             print(f"[WARN] Cannot check history: DB '{self.submissions_db_name}' not found.")
             return history

        # Use try-except for robustness
        try:
            conn = sqlite3.connect(self.submissions_db_name)
            cur = conn.cursor()

            # Get submissions for this specific slug
            cur.execute("""
                 SELECT status, language, submission_timestamp, runtime, memory
                 FROM submissions
                 WHERE title_slug=?
                 ORDER BY submission_timestamp DESC
            """, (problem_slug,))
            rows = cur.fetchall()
            if rows:
                 history["found"] = True
                 history["latest_status"] = rows[0][0] # Status of the most recent one
                 submission_details = []
                 for row_data in rows:
                     status, lang, ts, runtime, memory = row_data
                     try: # Attempt to format timestamp
                          dt = time.strftime('%Y-%m-%d %H:%M', time.localtime(int(ts)))
                     except (ValueError, TypeError, OverflowError): # Handle various potential errors
                          dt = ts # Keep original if conversion fails
                     submission_details.append({"status": status, "lang": lang, "time": dt, "runtime": runtime, "memory": memory})
                 history["submissions"] = submission_details

                 # Create a summary string
                 statuses = [s['status'] for s in history['submissions']]
                 status_counts = Counter(statuses)
                 history["details"] = f"Found {len(rows)} submissions. Latest: '{history['latest_status']}'. Statuses: {dict(status_counts)}."
                 print(f"[INFO] Found history for {problem_slug}: {history['details']}")
            else:
                  history["details"] = "No prior submissions found for this problem in the database."
                  print(f"[INFO] No submission history found for {problem_slug}.")


            # Also get problem details (difficulty, tags) from the problems table
            cur.execute("SELECT problem_name, difficulty, tags FROM problems WHERE title_slug=?", (problem_slug,))
            prob_data = cur.fetchone()
            if prob_data:
                 history["problem_name"] = prob_data[0] if prob_data[0] else problem_slug
                 history["difficulty"] = prob_data[1] if prob_data[1] else "Unknown"
                 try:
                     # Ensure tags are loaded correctly, default to empty list
                     history["tags"] = json.loads(prob_data[2]) if prob_data[2] and prob_data[2] != '[]' else []
                 except (json.JSONDecodeError, TypeError):
                     history["tags"] = []
                     print(f"[WARN] Failed to parse tags JSON for {problem_slug}: {prob_data[2]}")
            else:
                 # Problem metadata might not be in the DB if scraper failed or hasn't run for this problem
                 print(f"[WARN] Metadata for problem '{problem_slug}' not found in {self.submissions_db_name}.")
                 # Keep defaults set at the beginning of the function

        except sqlite3.Error as db_err:
             history["details"] = f"Database error checking problem history: {db_err}"
             print(f"[ERROR] Database error checking history for {problem_slug}: {db_err}")
        except Exception as e:
             history["details"] = f"Unexpected error checking problem history: {e}"
             print(f"[ERROR] Unexpected error checking history for {problem_slug}: {e}")
        finally:
             # Ensure connection is always closed
             if 'conn' in locals() and conn:
                  conn.close()
        return history

    # --------------------------------------------------------------------------
    # Method with ENHANCED prompt instructions for detailed LLM responses
    # --------------------------------------------------------------------------
    def answer_query(self, user_query, current_problem_url=None):
        """
        Generate an answer by constructing a detailed prompt with context and calling the LLM.
        The prompt instructions are enhanced to guide the LLM towards the user's preferred output style.
        """
        start_time = time.time()
        print(f"\n[INFO] answer_query started for query: '{user_query[:50]}...'")

        # --- 1. Gather Context ---
        problem_slug = None
        problem_context = {"details": "No specific problem URL provided."}
        problem_name = "N/A"
        problem_difficulty = "N/A"
        problem_tags = "N/A"
        problem_tags_list = [] # Keep list for potential pattern matching

        if current_problem_url:
            # Basic URL parsing, can be made more robust
            if isinstance(current_problem_url, str) and "/problems/" in current_problem_url:
                try:
                     problem_slug = current_problem_url.strip().rstrip("/").split("/problems/")[1].split("/")[0]
                     print(f"[INFO] Extracted slug: {problem_slug}")
                     problem_context = self.check_problem_history(problem_slug) # Fetches history & metadata
                     problem_name = problem_context.get("problem_name", problem_slug)
                     problem_difficulty = problem_context.get("difficulty", "Unknown")
                     problem_tags_list = problem_context.get("tags", [])
                     problem_tags = ", ".join(problem_tags_list) if problem_tags_list else "None"
                except IndexError:
                     problem_context = {"details": "Could not parse problem slug from URL."}
                     print(f"[WARN] Could not parse slug from URL: {current_problem_url}")
                except Exception as e:
                     problem_context = {"details": f"Error parsing URL or fetching history: {e}"}
                     print(f"[ERROR] Parsing URL/fetching history for {current_problem_url}: {e}")
            else:
                 problem_context = {"details": "Invalid or non-LeetCode URL provided."}
                 print(f"[INFO] Invalid or non-LeetCode URL: {current_problem_url}")


        # --- User Profile Context ---
        # Safely access profile data, providing defaults
        profile_data = self.mindset_profile if isinstance(self.mindset_profile, dict) else {}
        lang_counts = profile_data.get('language_counts', {})
        lang_summary = "N/A"
        preferred_lang = "Python" # Sensible default

        if isinstance(lang_counts, dict) and lang_counts:
             lang_summary = ", ".join([f"{lang} ({count})" for lang, count in lang_counts.items()])
             try:
                  preferred_lang = max(lang_counts, key=lang_counts.get)
             except ValueError: # Handle empty dict case
                  preferred_lang = "N/A"
        elif 'analysis_note' in profile_data: # Check if agent reported an issue
             lang_summary = profile_data['analysis_note']


        tag_counts = self.category_info if isinstance(self.category_info, dict) else {}
        tag_summary = "N/A"
        total_unique_tags = 0
        # Check if tag_counts is not empty and doesn't contain an error note
        if tag_counts and 'analysis_note' not in tag_counts:
             # Sort by count descending for top tags
             sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)
             top_tags = sorted_tags[:10] # Get top 10
             tag_summary = ", ".join([f"{tag} ({count})" for tag, count in top_tags])
             total_unique_tags = len(tag_counts)
        elif 'analysis_note' in tag_counts: # Check if agent reported an issue
            tag_summary = tag_counts['analysis_note']


        pattern_data = self.submission_patterns if isinstance(self.submission_patterns, dict) else {}
        pattern_notes = pattern_data.get("notes", ["Pattern analysis not available."])
        # Format notes for inclusion in prompt, indenting multi-line notes
        pattern_summary = "\n          - ".join(pattern_notes)


        # --- 2. Check if LLM is available ---
        if not self.llm_model:
             st.error("LLM Model not initialized. Cannot generate AI response. Check API Key.")
             context_summary_for_fallback = f"""
             **Your Coding Habits (Metadata-Based):**
             - Preferred Language (Inferred): {preferred_lang}
             - Language Frequency: {lang_summary}
             - Top Tags Encountered: {tag_summary} (Total unique: {total_unique_tags})
             - Submission Analysis Note: _{profile_data.get('analysis_note', 'N/A')}_
             - Inferred Submission Patterns:
               - {pattern_summary}

             **Context for Current Problem ({problem_name}):**
             - Difficulty: {problem_difficulty}
             - Tags: {problem_tags}
             - Your History: {problem_context.get('details', 'N/A')}
             """
             final_response_text = f"**DSA Assistant Response (LLM Disabled):**\n\n*LLM model is not available. Displaying basic context only.*\n{context_summary_for_fallback}\n\n**Your question:**\n```\n{user_query}\n```"
             print("[WARN] LLM disabled, returning basic context response.")
             return final_response_text, context_summary_for_fallback.strip()


        # --- 3. Construct Detailed LLM Prompt with ENHANCED Instructions ---
        # This context block is sent to the LLM
        context_for_prompt = f"""
        **User's LeetCode Profile Analysis Context:**
        - Preferred Language (Most Frequent): {preferred_lang}
        - Language Frequency Breakdown: {lang_summary}
        - General Submission Style Note: {profile_data.get('analysis_note', 'N/A')}
        - Top 10 DSA Tags Encountered by User: {tag_summary} (Total unique tags found in DB: {total_unique_tags})
        - Potential Learning Patterns (Inferred from non-accepted submissions):
          - {pattern_summary}

        **Current Problem Context (If URL Provided by User):**
        - Problem Name: {problem_name}
        - Problem Slug: {problem_slug or 'N/A'}
        - Difficulty: {problem_difficulty}
        - Official LeetCode Tags: {problem_tags or 'N/A'}
        - User's Submission History Summary for this Problem: {problem_context.get('details', 'N/A')}
        """

        # The user's query might contain code they want analyzed
        prompt = f"""
        You are an expert DSA Assistant and LeetCode Tutor AI. Your primary goal is to provide helpful, highly detailed, and pedagogical answers to the user's questions, leveraging their LeetCode profile context and the specific problem context when available.

        {context_for_prompt}

        **The User's Question is:**
        ```
        {user_query}
        ```

        **VERY IMPORTANT Instructions for Your Response (Strictly follow these steps, especially when the user provides code and a problem URL/context for analysis):**

        1.  **Acknowledge and Diagnose:** Directly address the user's core question (e.g., "what is the issue?", "how to optimize?", "explain concept?"). If the user provided code within their question, meticulously analyze it in the context of the specific LeetCode problem identified above (Name, Tags, Difficulty).

        2.  **Analyze User's Approach ("Instinct" vs. Intended):**
            * Identify the specific algorithm, data structures, and logic pattern implemented in the user's *provided code*. Name the pattern if possible (e.g., BFS, DFS, DP, Sliding Window, Vertical Traversal).
            * Compare this implemented approach to the standard, most efficient, or intended algorithm(s) for *this specific LeetCode problem* based on its constraints and typical solutions.
            * Explicitly state whether the user's approach is: (a) Suitable and correct, (b) Conceptually correct but inefficient, (c) Overly complex for the task, or (d) Fundamentally incorrect for solving *this specific problem*.
            * Provide a detailed explanation *why* the user's approach fits into one of these categories, referencing specifics from their code and the problem requirements. (e.g., "Your code uses a nested map keyed by level and vertical distance, characteristic of Vertical Order Traversal. However, 'Right Side View' only requires the last node per level in a standard BFS, making the nested map and vertical distance tracking unnecessary and overly complex for this specific task.").

        3.  **Provide Specific Test Cases & Edge Cases:**
            * Provide a bulleted list of important test cases specifically relevant to the identified LeetCode problem.
            * MUST include standard examples (e.g., a typical valid input).
            * MUST include critical **edge cases**. Examples: empty inputs (`[]`, `""`, `null root`), single-element inputs, inputs at boundary constraints (min/max values, max length), inputs designed to test specific logic (e.g., duplicates, negative numbers, all same values, specific tree shapes like skewed vs. complete, cycles in graphs if applicable).
            * Briefly explain *why* each edge case is important or what potential bug/issue it specifically tests for *this problem*.

        4.  **Code Assistance - Prioritize Explanation & Targeted Changes, then Full Solution:**
            * **Explain Errors/Inefficiencies Clearly:** Based on your diagnosis in step 2, detail the specific logical errors, bugs, performance issues, or conceptual mismatches found in the user's *provided code*. Reference line numbers or code snippets if helpful.
            * **Suggest Specific Changes (If Feasible):** If the user's code structure is reasonably close to a correct or more optimal solution, provide **specific, targeted code modifications**. Show the lines/blocks to change (using their preferred language: {preferred_lang}) and clearly explain the *reasoning* for each modification. Use diff format or before/after blocks if appropriate.
            * **Explain Need for Rewrite (If Necessary):** If the user's fundamental approach is incorrect, unsuitable, or significantly over-complicated (making targeted changes impractical), clearly explain *why* a rewrite using a different standard approach (e.g., standard BFS, iterative DP) is strongly recommended. Justify this recommendation based on clarity, correctness, and efficiency for *this problem*.
            * **Provide Full Correct Code:** After explaining the issues and the rationale for changes or a rewrite, provide the **full, clean, idiomatic, and well-commented correct code** for the standard/optimal solution in the user's preferred language ({preferred_lang}). Explain the key data structures and logic flow of this correct code step-by-step.

        5.  **Incorporate Learning Patterns (Use Carefully & Gently):**
            * Review the 'Potential Learning Patterns' context provided above (e.g., "TLE observed with tags like: dynamic-programming").
            * If a pattern seems directly relevant to the *current problem's tags* (Official LeetCode Tags listed above) AND relates to the *type of error diagnosed* in the user's code (e.g., the user's code is inefficient and the pattern mentions TLE for related tags), gently weave in a helpful tip.
            * Example: "As a general tip, for problems tagged with '{problem_tags_list[0] if problem_tags_list else 'this type'}', focusing on optimizing [e.g., time complexity] can be helpful, as suggested by the general pattern of [e.g., TLE] sometimes seen with these challenges. The [e.g., standard BFS] approach helps address this."
            * **Crucially:** Frame these as general observations/reminders based on broader patterns. Do *not* state that the user *definitely* made this mistake before unless the history summary is extremely specific. Connect it to the *problem type* and the *diagnosed issue* in their current code.

        6.  **Be Pedagogical & Encouraging:** Your primary goal is to help the user *learn and deeply understand* the concepts, algorithms, trade-offs, and potential pitfalls. Explain *why* things work or don't work. Use clear, precise language. Maintain a positive, supportive, and encouraging tone. Assume the user wants to learn, not just get the answer.

        7.  **Format for Readability:** Structure your response logically using Markdown effectively. Use headings (e.g., `### Diagnosis`), bullet points (`-`), numbered lists (`1.`), **bold text** for emphasis, and clearly formatted code blocks (```cpp ... ``` or ```python ... ```) for the user's preferred language ({preferred_lang}). Ensure the entire response is easy to read and follow.

        ---
        Assistant Answer (Provide your detailed, multi-part response below, following ALL applicable instructions above based on the user's query and the provided context):
        """ # End of the main f-string for prompt

        # --- 4. Call LLM API ---
        llm_response_text = "Error: Assistant could not generate a response." # Default error
        raw_llm_response = None
        try:
             print(f"[INFO] Sending request to Gemini API (using {self.llm_model._model_name})...")
             # print(f"--- PROMPT START (Truncated) ---\n{prompt[:1000]}...\n--- PROMPT END ---") # Debug: Print start of prompt
             # Use generate_content for the Gemini API
             response = self.llm_model.generate_content(prompt)
             raw_llm_response = response # Store for potential feedback analysis

             # Robustly check for response content
             if response.parts:
                  llm_response_text = response.text
                  print("[INFO] Received valid response from Gemini API.")
             else:
                  # Handle cases where the response might be blocked due to safety or other reasons
                  llm_response_text = "Error: Assistant response was blocked or empty."
                  print("[WARN] Gemini response was blocked or empty.")
                  # Attempt to get feedback if available
                  try:
                       feedback_info = response.prompt_feedback
                       llm_response_text += f"\nReason: {feedback_info}"
                       print(f"Prompt Feedback: {feedback_info}")
                  except Exception:
                       print("[WARN] Could not retrieve prompt feedback.")
                  # Check for finish reason if parts are missing
                  try:
                       finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                       llm_response_text += f"\nFinish Reason: {finish_reason}"
                       print(f"Finish Reason: {finish_reason}")
                  except Exception:
                       print("[WARN] Could not retrieve finish reason.")


        except Exception as e:
             print(f"[ERROR] Gemini API call failed: {e}")
             st.error(f"Failed to get response from LLM: {e}") # Show error in UI
             llm_response_text = f"**Error: Could not generate AI response.**\n*Details: {e}*"
             # Add feedback details if available from a partially failed response object
             if raw_llm_response and hasattr(raw_llm_response, 'prompt_feedback'):
                 try:
                      llm_response_text += f"\nPrompt Feedback: {raw_llm_response.prompt_feedback}"
                      print(f"Prompt Feedback during exception: {raw_llm_response.prompt_feedback}")
                 except Exception as feedback_e:
                      print(f"[WARN] Could not retrieve prompt feedback during exception: {feedback_e}")

        # --- 5. Format Final Output ---
        end_time = time.time()
        print(f"[INFO] answer_query finished in {end_time - start_time:.2f} seconds.")
        # Return the LLM response text and the context string used to generate it
        return llm_response_text, context_for_prompt.strip()


# --- Streamlit App Main Function ---
def run_app():
    # Set page config first
    st.set_page_config(layout="wide", page_title="DSA Assistant")

    st.title("🧠 DSA Assistant – Powered by Your LeetCode Data & Gemini")

    # Initialize session state variables robustly
    if 'scraper_active' not in st.session_state: st.session_state.scraper_active = False
    if 'assistant_agent' not in st.session_state: st.session_state.assistant_agent = None
    # Use separate flags for clarity on which data is loaded
    if 'data_loaded_submissions' not in st.session_state: st.session_state.data_loaded_submissions = False
    if 'data_loaded_solutions' not in st.session_state: st.session_state.data_loaded_solutions = False
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []


    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Controls & Status")
        st.divider()

        # --- Data Management Section ---
        st.subheader("Data Management")
        cookie_found = get_cookies() is not None
        if not cookie_found:
             st.warning("⚠️ `LEETCODE_SESSION_COOKIE` not found in `.env`. Scraping disabled.")
        else:
             st.success("✅ LeetCode session cookie found.")

        if not GOOGLE_API_KEY:
             st.warning("⚠️ `GOOGLE_API_KEY` not found in `.env`. AI responses disabled.")
        else:
             st.success("✅ Google API Key found.")

        # --- DB Status Checks ---
        # Display status of both databases
        db_status_placeholder = st.empty()
        sub_db_exists = os.path.exists(SUBMISSIONS_DB_NAME)
        sol_db_exists = os.path.exists(SOLUTIONS_DB_NAME)
        sub_count = 0
        sol_count = 0
        status_texts = []

        if sub_db_exists:
            try:
                conn_sub = sqlite3.connect(SUBMISSIONS_DB_NAME)
                # Check if tables exist before counting
                cur_sub = conn_sub.cursor()
                cur_sub.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='submissions'")
                submissions_table_exists = cur_sub.fetchone() is not None
                if submissions_table_exists:
                    sub_count = conn_sub.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
                    st.session_state.data_loaded_submissions = sub_count > 0
                    status_texts.append(f"Submissions DB ({sub_count} subs): {'✅ Loaded' if st.session_state.data_loaded_submissions else '⚠️ Empty'}")
                else:
                    status_texts.append("Submissions DB: ⚠️ Table missing")
                    st.session_state.data_loaded_submissions = False
                conn_sub.close()
            except Exception as e:
                 status_texts.append(f"Submissions DB: ❌ Error ({e})")
                 st.session_state.data_loaded_submissions = False
        else:
             status_texts.append("Submissions DB: ❌ Not Found")
             st.session_state.data_loaded_submissions = False

        if sol_db_exists:
             try:
                conn_sol = sqlite3.connect(SOLUTIONS_DB_NAME)
                # Check if table exists
                cur_sol = conn_sol.cursor()
                cur_sol.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='solutions'")
                solutions_table_exists = cur_sol.fetchone() is not None
                if solutions_table_exists:
                    sol_count = conn_sol.execute("SELECT COUNT(*) FROM solutions").fetchone()[0]
                    st.session_state.data_loaded_solutions = sol_count > 0
                    status_texts.append(f"Solutions DB ({sol_count} codes): {'✅ Loaded' if st.session_state.data_loaded_solutions else '⚠️ Empty'}")
                else:
                     status_texts.append("Solutions DB: ⚠️ Table missing")
                     st.session_state.data_loaded_solutions = False
                conn_sol.close()
             except Exception as e:
                 status_texts.append(f"Solutions DB: ❌ Error ({e})")
                 st.session_state.data_loaded_solutions = False
        else:
            status_texts.append("Solutions DB: ❌ Not Found")
            st.session_state.data_loaded_solutions = False

        db_status_placeholder.info(" | ".join(status_texts))


        # --- Scraper Button (Operates on SUBMISSIONS_DB_NAME) ---
        st.markdown("---")
        scrape_button_disabled = st.session_state.scraper_active or not cookie_found
        if st.button("🔄 Scrape LeetCode Submissions (API)", disabled=scrape_button_disabled, help="Fetches submission history into leetcode_submissions.db"):
            if cookie_found:
                st.session_state.scraper_active = True
                st.session_state.assistant_agent = None # Reset agent on rescrape
                st.session_state.data_loaded_submissions = False # Reset flag
                st.session_state.chat_history = [] # Clear chat history
                scrape_status = st.empty()
                progress_bar = st.progress(0.0, "Initializing...")
                try:
                    # Initialize the specific DB for submissions
                    if init_db(SUBMISSIONS_DB_NAME):
                        submissions = fetch_submissions_api(limit=20, progress_bar=progress_bar, status_placeholder=scrape_status)
                        progress_bar.empty() # Remove progress bar
                        if submissions:
                            stored_count, fetch_errors = store_data_api(submissions, db_name=SUBMISSIONS_DB_NAME, status_placeholder=scrape_status)
                            # Verify data loaded state after storage
                            conn_check = sqlite3.connect(SUBMISSIONS_DB_NAME)
                            sub_count_check = conn_check.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
                            conn_check.close()
                            if sub_count_check > 0:
                                 st.session_state.data_loaded_submissions = True
                                 final_msg = f"✅ Scrape complete. Stored {stored_count} new. Total Subs in DB: {sub_count_check}."
                                 if fetch_errors > 0: final_msg += f" ({fetch_errors} metadata errors)"
                                 scrape_status.success(final_msg)
                            else:
                                scrape_status.warning("⚠️ Scraping ran, but Submissions DB is still empty.")
                                st.session_state.data_loaded_submissions = False
                        else:
                            scrape_status.warning("⚠️ No submissions were fetched. Check LeetCode cookie or network.")
                            # Re-check if DB had old data just in case
                            try:
                                conn_check = sqlite3.connect(SUBMISSIONS_DB_NAME)
                                sub_count_check = conn_check.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
                                conn_check.close()
                                st.session_state.data_loaded_submissions = sub_count_check > 0
                            except: pass # Ignore check errors here
                    else:
                         scrape_status.error("❌ Failed to initialize Submissions database.")
                         st.session_state.data_loaded_submissions = False
                finally:
                    st.session_state.scraper_active = False
                    st.rerun() # Rerun to update UI and potentially initialize agent
            else:
                 st.error("Cannot scrape without LEETCODE_SESSION_COOKIE.")

        st.info(f"**Scraper:** Updates `{SUBMISSIONS_DB_NAME}`\n**Code Analysis:** Uses `{SOLUTIONS_DB_NAME}`\n(Run code conversion script separately for code analysis)")


        st.divider()
        # --- Agent Status Section ---
        st.subheader("Assistant Status")
        agent_status_placeholder = st.empty()

        # Conditions for initializing the main DSAAssistantAgent
        # Need API key AND at least one relevant database loaded
        has_necessary_data = st.session_state.data_loaded_submissions or st.session_state.data_loaded_solutions
        can_initialize_agent = has_necessary_data and GOOGLE_API_KEY
        should_initialize_agent = can_initialize_agent and st.session_state.assistant_agent is None

        if should_initialize_agent:
             agent_status_placeholder.info("⏳ Initializing Assistant Agent...")
             print("[INFO] Conditions met. Initializing Assistant Agent...")
             try:
                 # Initialize the main agent; it handles initializing sub-agents internally
                 st.session_state.assistant_agent = DSAAssistantAgent(
                     submissions_db_name=SUBMISSIONS_DB_NAME,
                     solutions_db_name=SOLUTIONS_DB_NAME
                 )
                 # Check if LLM model loaded successfully within the agent
                 if st.session_state.assistant_agent.llm_model:
                     # Check status of sub-agents after initialization
                     status_msg = "✅ Assistant is ready."
                     if not st.session_state.assistant_agent.mindset_agent: status_msg += " (Mindset Analysis Limited - No Code DB?)"
                     if not st.session_state.assistant_agent.category_agent: status_msg += " (Category Analysis Limited - No Submissions DB?)"
                     agent_status_placeholder.success(status_msg)
                     print(f"[INFO] Assistant Agent initialized successfully. Status: {status_msg}")
                     time.sleep(0.5) # Short delay before rerun
                     st.rerun() # Rerun to update main page instantly after successful init
                 else:
                     # Agent object created, but LLM failed
                     agent_status_placeholder.warning("⚠️ Assistant object created, but LLM failed. AI responses disabled.")
                     print("[WARN] Assistant Agent initialized, but LLM failed.")

             except Exception as e:
                  agent_status_placeholder.error(f"❌ Agent initialization failed: {e}")
                  print(f"[ERROR] Agent initialization failed: {e}")
                  st.session_state.assistant_agent = None # Ensure agent is None on failure

        # Display current status if agent exists or based on other states
        elif st.session_state.assistant_agent is not None:
             current_status = "✅ Assistant is active."
             if not st.session_state.assistant_agent.llm_model: current_status = "⚠️ Assistant active, but LLM failed."
             # Add sub-agent status
             if not st.session_state.assistant_agent.mindset_agent: current_status += " (Mindset Limited)"
             if not st.session_state.assistant_agent.category_agent: current_status += " (Category Limited)"
             agent_status_placeholder.success(current_status) # Use success even if limited, as object exists
        elif st.session_state.scraper_active:
             agent_status_placeholder.info("⏳ Scraper running...")
        elif not GOOGLE_API_KEY:
            agent_status_placeholder.error("❌ Cannot initialize assistant: Google API Key missing.")
        elif not has_necessary_data:
             agent_status_placeholder.warning("⚠️ Load data to enable assistant (Run Scraper / Code Converter).")
        else:
             # Should typically be initializing if conditions met, but display info otherwise
              agent_status_placeholder.info("ℹ️ Assistant not yet initialized.")


    # --- Main Chat Interface Area ---
    st.header("💬 Chat with your DSA Assistant")
    st.markdown("""
    Ask questions about DSA concepts, specific LeetCode problems (paste URL for context),
    or paste your code and ask for analysis (e.g., "What's the issue with this code?", "How can I optimize this?").
    """)
    st.divider()

    # Use columns for layout
    col1, col2 = st.columns([2, 1]) # Input area larger

    with col1:
        st.subheader("Your Query")
        current_problem_url = st.text_input(
            "🔗 LeetCode Problem URL (Optional, for context):",
            placeholder="e.g., https://leetcode.com/problems/two-sum/"
        )
        user_query = st.text_area(
            "❓ Your Question / Code to Analyze:",
            placeholder="e.g., Explain BFS vs DFS.\nOr paste your code for the problem above and ask:\nWhat is the issue with this code?",
            height=200
        )

        # Determine if the assistant is fully ready (agent object exists AND LLM is loaded)
        is_assistant_ready = (
            st.session_state.assistant_agent is not None and
            st.session_state.assistant_agent.llm_model is not None
        )
        submit_button_disabled = not user_query or st.session_state.scraper_active or not is_assistant_ready

        if st.button("🚀 Get Answer", type="primary", disabled=submit_button_disabled, use_container_width=True):
            if is_assistant_ready:
                 with st.spinner("🤖 Assistant is thinking (using Gemini)..."):
                     try:
                         # answer_query now returns response_text, context_text
                         response_text, context_text = st.session_state.assistant_agent.answer_query(user_query, current_problem_url)
                         # Add to chat history (newest first)
                         st.session_state.chat_history.insert(0, (user_query, current_problem_url, response_text, context_text))
                         st.rerun() # Rerun to display the new message in history
                     except Exception as e:
                          st.error(f"An error occurred while getting the answer: {e}")
                          print(f"[ERROR] An error occurred in answer_query call or processing its result: {e}")
            else:
                 # Give more specific feedback if button was somehow enabled when not ready
                 if not GOOGLE_API_KEY:
                      st.error("Cannot get answer: Google API Key is missing.")
                 elif not (st.session_state.data_loaded_submissions or st.session_state.data_loaded_solutions):
                      st.error("Cannot get answer: Load data first (Scraper/Converter).")
                 elif st.session_state.assistant_agent is None:
                      st.error("Cannot get answer: Assistant not initialized.")
                 elif st.session_state.assistant_agent.llm_model is None:
                      st.error("Cannot get answer: LLM initialization failed.")
                 else:
                      st.error("Cannot get answer: Assistant is not ready for an unknown reason.")


    with col2:
        st.subheader("💡 Assistant Status Summary")
        # Show summary based on agent state
        if is_assistant_ready:
            st.success("Ready to Chat!")
            try:
                # Display key metrics if available
                profile_data = st.session_state.assistant_agent.mindset_profile
                lang_counts = profile_data.get('language_counts', {})
                if lang_counts and isinstance(lang_counts, dict):
                     preferred_lang = max(lang_counts, key=lang_counts.get, default="N/A")
                     st.metric("Your Top Language", preferred_lang)
                elif 'analysis_note' in profile_data:
                    st.info(f"Language Analysis: {profile_data['analysis_note']}")
                else:
                    st.info("Run code conversion script + restart for language analysis.")

                cat_info = st.session_state.assistant_agent.category_info
                if cat_info and isinstance(cat_info, dict) and 'analysis_note' not in cat_info:
                    st.metric("Unique Tags Encountered", len(cat_info))
                elif 'analysis_note' in cat_info:
                    st.info(f"Category Analysis: {cat_info['analysis_note']}")
                else:
                    st.info("Run scraper + restart for tag analysis.")
            except Exception as e:
                 st.warning(f"Error displaying metrics: {e}")

        elif st.session_state.assistant_agent is not None and st.session_state.assistant_agent.llm_model is None:
             st.error("LLM Error - AI Disabled.")
             st.info("Check API Key & Logs.")
        elif not (st.session_state.data_loaded_submissions or st.session_state.data_loaded_solutions):
             st.warning("Load Data.")
             st.info("Run Scraper &/or Code Converter script via sidebar.")
        elif not GOOGLE_API_KEY:
             st.error("API Key Missing.")
             st.info("Set GOOGLE_API_KEY in .env file.")
        else:
             st.info("Initializing...") # Fallback status


    st.divider()

    # --- Display Chat History ---
    st.header("📜 Conversation History")
    if not st.session_state.chat_history:
        st.info("No questions asked yet in this session.")
    else:
        # Display history newest first
        for i, (q_query, q_url, q_response, q_context) in enumerate(st.session_state.chat_history):
             # Use a container with border for each interaction
             with st.container(border=True):
                st.caption(f"Interaction {len(st.session_state.chat_history) - i}") # Numbering
                # Display user query using markdown for potential code formatting
                st.markdown("**You Asked:**")
                st.markdown(f"```\n{q_query}\n```") # Display query in code block
                if q_url:
                    st.markdown(f"*Problem Context URL:* `{q_url}`")
                st.markdown("**Assistant Responded:**")
                st.markdown(q_response, unsafe_allow_html=False) # Render assistant's markdown response
                # Expander to view the context sent to the LLM for debugging/transparency
                with st.expander("View Context Sent to LLM for this Response"):
                     st.text_area(f"Context_{i}", q_context, height=250, disabled=True, label_visibility="collapsed")
             # Add a visual separator between history items
             # st.divider() # Optional: uncomment for more separation

# --- Main Execution Guard ---
if __name__ == "__main__":
    print("========================================")
    print("Starting Streamlit DSA Assistant App...")
    print(f"Submissions DB: {SUBMISSIONS_DB_NAME}")
    print(f"Solutions DB: {SOLUTIONS_DB_NAME}")
    print(f"Google API Key Loaded: {GOOGLE_API_KEY is not None}")
    print(f"LeetCode Cookie Loaded: {LEETCODE_SESSION_COOKIE is not None}")
    print("========================================")
    run_app()
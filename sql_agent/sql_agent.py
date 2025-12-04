"""
SQL Knowledge Base Agent
A free alternative to Copilot Studio for SQL query generation

This agent reads SQL files AND CSV/XLS files from a knowledge base folder
and uses Claude API to help generate SQL queries based on user questions.
"""

import os
import anthropic
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

class SQLAgent:
    def __init__(self, api_key, knowledge_base_folder="sql_knowledge_base"):
        """
        Initialize the SQL Agent
        
        Args:
            api_key: Your Anthropic API key
            knowledge_base_folder: Folder containing your SQL/CSV/XLS files
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.knowledge_base_folder = Path(knowledge_base_folder)
        self.knowledge_base = {}      # stores .sql files
        self.csv_tables = {}          # stores loaded CSV/XLS tables
        self.load_knowledge_base()
        self.load_data_files()        # NEW: load csv/xlsx files

    # ----------------------------------------------------------------------
    # Load SQL files
    # ----------------------------------------------------------------------
    def load_knowledge_base(self):
        """Load all SQL files from the knowledge base folder"""
        if not self.knowledge_base_folder.exists():
            self.knowledge_base_folder.mkdir(parents=True)
            print(f"Created knowledge base folder: {self.knowledge_base_folder}")
            return
        
        sql_files = list(self.knowledge_base_folder.glob("**/*.sql"))
        
        if not sql_files:
            print(f"No SQL files found in {self.knowledge_base_folder}")
        else:
            for sql_file in sql_files:
                try:
                    with open(sql_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.knowledge_base[sql_file.name] = content
                        print(f"✓ Loaded SQL file: {sql_file.name}")
                except Exception as e:
                    print(f"✗ Error loading {sql_file.name}: {e}")
        
        print(f"Total SQL files loaded: {len(self.knowledge_base)}\n")

    # ----------------------------------------------------------------------
    # Load CSV / Excel files
    # ----------------------------------------------------------------------
    def load_data_files(self):
        """Load CSV, XLS, XLSX files into Pandas and infer schema"""
        data_files = list(self.knowledge_base_folder.glob("**/*.csv")) + \
                     list(self.knowledge_base_folder.glob("**/*.xlsx")) + \
                     list(self.knowledge_base_folder.glob("**/*.xls"))

        if not data_files:
            print("No CSV/XLS files found in knowledge base.")
            return

        print("📦 Loading CSV/XLS files...")

        for file in data_files:
            try:
                if file.suffix.lower() == ".csv":
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                self.csv_tables[file.stem] = df
                print(f"✓ Loaded table: {file.stem} ({len(df)} rows)")
            except Exception as e:
                print(f"✗ Failed to load {file.name}: {e}")

        print(f"Total CSV/XLS tables loaded: {len(self.csv_tables)}\n")

    # ----------------------------------------------------------------------
    # Build context for Claude
    # ----------------------------------------------------------------------
    def get_context(self):
        """Combine SQL files + CSV/XLS schema context"""
        context = "=== SQL KNOWLEDGE BASE ===\n\n"

        # Add SQL file contents
        for filename, content in self.knowledge_base.items():
            context += f"--- File: {filename} ---\n{content}\n\n"

        # Add CSV/XLS schemas
        context += "\n=== DATA FILE SCHEMAS (CSV / XLS) ===\n\n"

        if not self.csv_tables:
            context += "(No CSV/XLS tables loaded)\n"
        else:
            for table_name, df in self.csv_tables.items():
                context += f"Table: {table_name}\n"
                context += "Columns:\n"
                for col in df.columns:
                    context += f"  - {col} ({df[col].dtype})\n"
                context += f"Row count: {len(df)}\n\n"

        return context

    # ----------------------------------------------------------------------
    # Ask Claude to generate SQL logic
    # ----------------------------------------------------------------------
    def ask(self, question, include_explanation=True):
        context = self.get_context()

        system_prompt = """You are an expert SQL and data analysis assistant. 
You have access to:
- SQL files written by the user
- CSV and Excel tables that have been loaded into memory

Your job:
1. Understand the question
2. Use SQL-like logic to analyze the CSV/XLS data
3. Generate queries referencing table names exactly as shown
4. Give clean and correct SQL (even though data is CSV)
5. Provide explanation unless the user requests otherwise
"""

        user_prompt = f"""{context}

USER QUESTION:
{question}

Please generate the SQL query needed to answer the question.
{'Include explanation.' if include_explanation else 'Do NOT include explanation.'}
"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return message.content[0].text
        
        except Exception as e:
            return f"Error: {str(e)}"

    # ----------------------------------------------------------------------
    # Reload everything
    # ----------------------------------------------------------------------
    def reload_knowledge_base(self):
        print("\n🔄 Reloading knowledge base...")
        self.knowledge_base.clear()
        self.csv_tables.clear()
        self.load_knowledge_base()
        self.load_data_files()   # reload CSV/XLS
        print("Reload complete!")

# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SQL KNOWLEDGE BASE AGENT (SQL + CSV + XLS Support)")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ API key not found! Add it to .env file.")
        return

    agent = SQLAgent(api_key)

    if not agent.knowledge_base and not agent.csv_tables:
        print("\n📁 Add SQL/CSV/XLS files to the 'sql_knowledge_base' folder")
        print("Then run this script again!")
        return

    print("\nAgent ready! Ask your questions or type:")
    print("  reload  - reload files")
    print("  list    - show loaded files & tables")
    print("  exit    - quit\n")

    while True:
        question = input("💬 You: ").strip().lower()

        if question == "exit":
            print("Goodbye! 👋")
            break

        if question == "reload":
            agent.reload_knowledge_base()
            continue

        if question == "list":
            print("\n📚 SQL Files:", list(agent.knowledge_base.keys()))
            print("📦 Tables:", list(agent.csv_tables.keys()))
            continue

        print("\n🤖 Agent:")
        print(agent.ask(question))


if __name__ == "__main__":
    main()

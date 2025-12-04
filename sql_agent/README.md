# SQL Knowledge Base Agent
## Your FREE Alternative to Copilot Studio

This Python-based SQL agent uses Claude API to analyze your SQL files and generate queries - completely replacing your Copilot Studio agent at a fraction of the cost!

---

## 🚀 Quick Start

### Step 1: Get Your Claude API Key (FREE Credits Available!)

1. Go to: https://console.anthropic.com/
2. Sign up for an account
3. You'll get **FREE API credits** to start ($5 credit = thousands of queries!)
4. Go to "API Keys" and create a new key
5. Copy your API key

### Step 2: Install Python Requirements

```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install anthropic
```

### Step 3: Set Up Your API Key

**Option A: Environment Variable (Recommended)**
```bash
# Windows
set ANTHROPIC_API_KEY=your_api_key_here

# Mac/Linux
export ANTHROPIC_API_KEY=your_api_key_here
```

**Option B: Enter When Prompted**
The script will ask for your API key if it's not set as an environment variable.

### Step 4: Add Your SQL Files

1. Create a folder called `sql_knowledge_base` (script creates this automatically)
2. Put all your SQL files (.sql) in this folder
3. The agent will automatically load them

### Step 5: Run the Agent

```bash
python sql_agent.py
```

---

## 💡 Usage Examples

### Example 1: Generate Query from Scratch
```
You: How do I get all employees hired in 2024?

Agent: Here's a SQL query to get all employees hired in 2024:

```sql
SELECT 
    employee_id,
    first_name,
    last_name,
    hire_date,
    department
FROM employees
WHERE YEAR(hire_date) = 2024
ORDER BY hire_date;
```

This query filters employees where the year of their hire_date is 2024...
```

### Example 2: Reference Your Knowledge Base
```
You: Based on my existing queries, how can I join employee and department tables?

Agent: Looking at your knowledge base file "employee_queries.sql", 
I can see you typically use LEFT JOIN. Here's the pattern:

```sql
SELECT 
    e.employee_id,
    e.first_name,
    d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;
```
```

### Example 3: Complex Analysis
```
You: Create a query to find the top 5 salespeople by revenue this quarter

Agent: [Generates complex query with CTEs, window functions, etc.]
```

---

## 📁 Organizing Your SQL Knowledge Base

### Recommended Structure:

```
sql_knowledge_base/
├── common_queries.sql          # Your frequently used queries
├── table_definitions.sql       # CREATE TABLE statements
├── views.sql                   # Your custom views
├── stored_procedures.sql       # Stored procedures
├── complex_joins.sql           # Join patterns you use often
└── data_analysis.sql           # Analysis queries
```

### Example SQL File Format:

**common_queries.sql:**
```sql
-- Get all active employees
-- Description: Returns employees with active status
SELECT employee_id, first_name, last_name, hire_date
FROM employees
WHERE status = 'Active'
ORDER BY hire_date DESC;

-- Get department summary
-- Description: Count of employees per department
SELECT 
    d.department_name,
    COUNT(e.employee_id) as employee_count
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
GROUP BY d.department_name;
```

**💡 Tip:** Add comments to your SQL files! The agent uses these to understand context.

---

## 🔧 Commands

While the agent is running, you can use these commands:

- **`reload`** - Reload SQL files after adding new ones (no need to restart!)
- **`list`** - Show all loaded SQL files
- **`exit`** - Quit the agent

---

## 💰 Cost Comparison

| Service | Cost | Your Usage |
|---------|------|------------|
| **Copilot Studio** | $200+/month | Enterprise only |
| **Claude API** | ~$0.003 per query | Pay only for what you use |
| **This Agent** | **FREE** to start | $5 credit = ~1,500 queries |

### Example Monthly Costs:
- 100 queries/month: **~$0.30**
- 500 queries/month: **~$1.50**
- 1000 queries/month: **~$3.00**

**Much cheaper than Copilot Studio!**

---

## ⚡ Advanced Usage

### Use in Your Own Scripts

```python
from sql_agent import SQLAgent

# Initialize
agent = SQLAgent(api_key="your_key", knowledge_base_folder="my_sql_files")

# Ask questions
response = agent.ask("How do I calculate year-over-year growth?")
print(response)

# Reload after adding files
agent.reload_knowledge_base()
```

### Add to Your Data Pipeline

You can integrate this into Azure Data Factory, Databricks, or any Python workflow:

```python
import sql_agent

# Generate query based on parameters
query = agent.ask(f"Create a query to extract {table_name} data for {date_range}")

# Execute in your database
cursor.execute(query)
```

---

## 🎯 Perfect For Your TMF Work

This agent can help you with:

✅ **MDP Queries** - Generate complex data extraction queries  
✅ **Viewpoint Analysis** - Create analytical SQL for reporting  
✅ **Enate Data** - Combine queries from factTickets, factCases, factActions  
✅ **Power BI Backends** - Generate SQL for your dashboards  
✅ **Azure Projects** - Quick query generation for Databricks/SQL Server  

---

## 🆘 Troubleshooting

### "No module named 'anthropic'"
```bash
pip install anthropic
```

### "No SQL files found"
Make sure your .sql files are in the `sql_knowledge_base` folder

### "API key error"
Check your API key is correct and has credits available at console.anthropic.com

### Need to add new files?
Just drop them in `sql_knowledge_base` folder and type `reload` - no restart needed!

---

## 🚀 Next Steps

1. **Start small**: Add 2-3 of your most common SQL queries
2. **Test it out**: Ask questions and see how it performs
3. **Expand gradually**: Add more queries as you use it
4. **Share with team**: They can use it too (each person needs their own API key)

---

## 📝 Notes

- **Privacy**: Your SQL files stay on your computer. Only your questions and file contents are sent to Claude API
- **Security**: Never put actual passwords or sensitive data in your SQL files
- **Learning**: The more SQL examples you add, the better it understands your patterns
- **Updates**: You can modify `sql_agent.py` to customize behavior

---

## ❓ Questions?

This is YOUR agent now. You can:
- Modify the code however you want
- Add features (save responses, export queries, etc.)
- Integrate it into your workflows
- Share with your team

**Need help?** Just ask Claude (me!) and I can help you customize it further!

---

**Enjoy your FREE SQL Agent! 🎉**

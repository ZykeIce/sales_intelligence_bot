# Sales Intelligence Bot

## Overview
This project is a Sales Intelligence Bot designed to analyze thousands of companies and their websites to determine:
- If they are hiring (using web, LinkedIn, and job board signals)
- The probability (%) that they will buy your product
- The reasoning and factors behind the AI's decision

## Features
- Scrapes company websites, LinkedIn, and job boards for hiring and growth signals
- Uses an AI model to analyze all gathered data and output a purchase probability and detailed reasoning
- Saves results in timestamped session folders, with support for resuming and custom company ranges
- Outputs concise terminal summaries and saves all detailed reasoning for later review

## Setup
1. **Clone the repository**
2. **Install dependencies** (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure your product description and OpenAI API key** in `config.py`:
   - Set `PRODUCT_DESCRIPTION` to your actual product
   - Set `OPENAI_API_KEY` to your OpenAI key

4. **Prepare your company data** in `data/companies.csv` (columns: `domain_name`, `company_name`, `website`, `no_of_employees`, ...)

## Usage
Run the bot with:
```bash
python main.py [--start X] [--end Y] [--version Z]
```
- `--start X` : Start index (inclusive) in the company list (default: 0)
- `--end Y`   : End index (exclusive) in the company list (default: end of file)
- `--version Z` : Version string for the session folder (default: 1.0)

**Example:**
```bash
python main.py --start 100 --end 200 --version 1.1
```
This will process companies 100 to 199 and save results in a folder like `results/session_YYYYMMDD_HHMMSS_from_100_to_200_v1.1`.

## Workflow
1. **Scraping:**
   - The bot explores the homepage and all relevant subpages (Careers, About, Team, News, Blog, etc.)
   - It also checks LinkedIn and job boards for hiring signals
2. **AI Analysis:**
   - All gathered data and your product description are sent to the AI model
   - The AI outputs a purchase probability, reasoning, open roles, growth signals, product fit, and red flags
3. **Output:**
   - Results are saved in CSV, JSON, and text files in a session folder
   - Terminal output is concise (summary per company)
   - All detailed reasoning is saved for later review

## Output Files
Each session folder contains:
- `results.csv` : Basic results (summary per company)
- `detailed_results.json` : Full structured data for each company
- `scraped_texts.txt` : All scraped text and AI analysis
- `status.json` : Session status and progress
- `realtime_log.txt` : Real-time processing log

## Resuming
- You can stop the script at any time (Ctrl+C)
- To resume, simply run the script again with the same range/version; it will skip already-processed companies

## Customization
- You can adjust scraping depth, page types, and AI prompt in `scraper.py` and `ai_model.py`
- For advanced use (parallel runs, custom company lists, etc.), see the code comments

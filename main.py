import data_processor
import scraper
import ai_model
import pandas as pd
from tqdm import tqdm
import time
import json
import os
import signal
import sys
from datetime import datetime
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global variables for graceful shutdown
running = True
session_folder = None
VERSION = '1.0'

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print(f"\n\n🛑 Received stop signal. Saving progress and shutting down gracefully...")
    running = False

def create_session_folder(start=0, end=None, version=VERSION):
    """Create a timestamped session folder for results, including company range and version."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"session_{timestamp}_from_{start}_to_{end if end is not None else 'end'}_v{version}"
    session_folder = os.path.join('results', folder_name)
    os.makedirs(session_folder, exist_ok=True)
    return session_folder, timestamp

def load_existing_results(session_folder):
    """Load existing results if resuming"""
    csv_file = os.path.join(session_folder, "results.csv")
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        print(f"Found existing results: {len(existing_df)} companies already processed")
        return existing_df
    return None

def save_progress_realtime(session_folder, results, detailed_results, timestamp, current_company=""):
    """Save current progress in real-time"""
    # Save basic results to CSV
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(session_folder, "results.csv")
    results_df.to_csv(csv_file, index=False)
    
    # Save detailed results to JSON
    json_file = os.path.join(session_folder, "detailed_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Save current progress to a status file
    status_file = os.path.join(session_folder, "status.json")
    status_info = {
        "session_timestamp": timestamp,
        "total_processed": len(results),
        "last_updated": datetime.now().isoformat(),
        "current_company": current_company,
        "files": {
            "csv": csv_file,
            "json": json_file
        }
    }
    
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(status_info, f, indent=2)
    
    # Append to real-time log file
    log_file = os.path.join(session_folder, "realtime_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {len(results)} companies. Current: {current_company}\n")

def append_to_text_file(session_folder, result):
    """Append a single result to the text file"""
    text_file = os.path.join(session_folder, "scraped_texts.txt")
    with open(text_file, 'a', encoding='utf-8') as f:
        f.write(f"=== {result['company_name']} ({result['website_url']}) ===\n")
        f.write(f"Hiring: {'Yes' if result['is_hiring'] else 'No'}\n")
        f.write(f"Hiring Reasoning: {result['hiring_reasoning']}\n")
        if result['additional_sources']:
            f.write(f"Additional Sources: {result['additional_sources']}\n")
        f.write(f"Employee Count: {result['employee_count']}\n")
        f.write(f"AI Reasoning: {result['ai_reasoning']}\n")
        f.write(f"Purchase Probability %: {result['purchase_probability_percent']}\n")
        f.write(f"Raw AI Analysis:\n{result['ai_analysis_raw']}\n")
        f.write(f"Scraped Text:\n{result['scraped_text']}\n")
        f.write("\n" + "="*80 + "\n\n")

def parse_ai_analysis(ai_analysis):
    reasoning = ""
    percentage = "N/A"
    if ai_analysis:
        match = re.search(r'PERCENTAGE:\s*([0-9]{1,3}%)', ai_analysis, re.IGNORECASE)
        if match:
            percentage = match.group(1).strip()
            reasoning = ai_analysis.replace(match.group(0), "").strip()
        else:
            # fallback: try to find any percentage
            pct_match = re.search(r'([0-9]{1,3}%)', ai_analysis)
            if pct_match:
                percentage = pct_match.group(1)
            reasoning = ai_analysis
    return reasoning, percentage

def process_company(row, session_folder, timestamp, total_companies):
    import time
    company_start = time.time()
    company_name = row.company_name
    website_url = row.website
    domain_name = row.domain_name
    employee_count = getattr(row, 'no_of_employees', 'N/A')

    # Scrape website
    is_hiring, hiring_reasoning, scraped_text, additional_info = scraper.scrape_website(website_url)
    if scraped_text:
        ai_analysis = ai_model.get_purchase_probability(scraped_text)
        reasoning, percentage = parse_ai_analysis(ai_analysis)
    else:
        ai_analysis = "Could not extract text from website"
        reasoning = ai_analysis
        percentage = "N/A"

    company_time = time.time() - company_start
    result = {
        "Company Name": company_name,
        "Domain": domain_name,
        "Website": website_url,
        "Employee Count": employee_count,
        "Hiring?": "Yes" if is_hiring else "No",
        "Hiring Reasoning": hiring_reasoning,
        "Additional Sources": additional_info,
        "AI Reasoning": reasoning,
        "Purchase Probability %": percentage,
        "detailed": {
            "company_name": company_name,
            "domain_name": domain_name,
            "website_url": website_url,
            "employee_count": employee_count,
            "is_hiring": is_hiring,
            "hiring_reasoning": hiring_reasoning,
            "additional_sources": additional_info,
            "scraped_text": scraped_text,
            "ai_reasoning": reasoning,
            "purchase_probability_percent": percentage,
            "ai_analysis_raw": ai_analysis,
            "analysis_timestamp": datetime.now().isoformat()
        },
        "company_time": company_time
    }
    # Save scraped_texts.txt immediately (thread-safe, append-only)
    append_to_text_file(session_folder, result['detailed'])
    return result

def main():
    """
    Main function to run the sales intelligence bot with real-time saving.
    """
    global running, session_folder
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check if product description is set
    if ai_model.config.PRODUCT_DESCRIPTION == "YOUR_PRODUCT_DESCRIPTION_HERE":
        print("ERROR: Please update PRODUCT_DESCRIPTION in config.py with your actual product information!")
        return
    
    # Parse command-line arguments for custom range
    parser = argparse.ArgumentParser(description="Sales Intelligence Bot")
    parser.add_argument('--start', type=int, default=0, help='Start index (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
    parser.add_argument('--version', type=str, default=VERSION, help='Session version string')
    args = parser.parse_args()
    
    # Create session folder with range and version
    session_folder, timestamp = create_session_folder(args.start, args.end, args.version)
    print(f"Session folder: {session_folder}")
    print(f"📁 Results will be saved in real-time to: {session_folder}")
    print(f"⏹️  Press Ctrl+C to stop and save progress")
    print(f"🔄 You can resume later by running the script again")
    
    # Load companies
    print("\nLoading company data...")
    companies_df = data_processor.load_companies("data/companies.csv")
    print(f"Loaded {len(companies_df)} companies")

    # Apply custom range if specified
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(companies_df)
    companies_df = companies_df.iloc[start_idx:end_idx]
    
    # Check for existing results to resume
    existing_results = load_existing_results(session_folder)
    
    if existing_results is not None:
        # Resume from where we left off
        processed_companies = set(existing_results['Company Name'].tolist())
        companies_to_process = companies_df[~companies_df['company_name'].isin(processed_companies)]
        results = existing_results.to_dict('records')
        print(f"🔄 Resuming: {len(companies_to_process)} companies remaining to process")
    else:
        # Start fresh
        companies_to_process = companies_df  # Now using the custom range
        results = []
        print(f"🚀 Starting fresh: processing first {len(companies_to_process)} companies")

    detailed_results = []
    
    print(f"\n{'='*60}")
    print("Starting analysis... (Press Ctrl+C to stop anytime)")
    print(f"{'='*60}")
    
    max_workers = 8  # Tune this for your system/API limits
    results = []
    detailed_results = []
    start_time = time.time()
    company_count = 0
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in companies_to_process.itertuples(index=False):
                futures.append(executor.submit(process_company, row, session_folder, timestamp, len(companies_to_process)))
            for future in as_completed(futures):
                result = future.result()
                results.append({k: v for k, v in result.items() if k != 'detailed' and k != 'company_time'})
                detailed_results.append(result['detailed'])
                company_count += 1
                # Print summary for this company
                print(f"\n{'='*60}")
                print(f"[{company_count}] {result['Company Name']} ({result['Website']})")
                print(f"  Hiring: {result['Hiring?']} | Probability: {result['Purchase Probability %']} | Time: {result['company_time']:.1f}s")
                if result['Purchase Probability %'] == "N/A":
                    print(f"⚠️  Warning: AI did not provide a percentage!")
                print(f"{'='*60}")
                # Save progress in real-time (main thread)
                save_progress_realtime(session_folder, results, detailed_results, timestamp, result['Company Name'])
        total_time = time.time() - start_time
        # Final save
        if results:
            save_progress_realtime(session_folder, results, detailed_results, timestamp, "COMPLETED")
            print(f"\n{'='*60}")
            print(f"📊 Analysis Summary")
            print(f"{'='*60}")
            print(f"📁 Session folder: {session_folder}")
            print(f"📄 Files saved:")
            print(f"  - results.csv (basic results in CSV format)")
            print(f"  - detailed_results.json (complete structured data)")
            print(f"  - scraped_texts.txt (all scraped text and analysis)")
            print(f"  - status.json (session status and progress)")
            print(f"  - realtime_log.txt (real-time processing log)")
            print(f"📈 Total companies processed: {len(results)}")
            print(f"⏱️  Total session time: {total_time:.1f} seconds")
            # Show summary
            results_df = pd.DataFrame(results)
            hiring_count = len(results_df[results_df['Hiring?'] == 'Yes'])
            print(f"💼 Companies hiring: {hiring_count}")
            print(f"❌ Companies not hiring: {len(results_df) - hiring_count}")
            if not running:
                print(f"\n🔄 You can resume by running the script again - it will continue from where it left off!")
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted by user. Saving final progress...")
        save_progress_realtime(session_folder, results, detailed_results, timestamp, "INTERRUPTED")

if __name__ == "__main__":
    main()

import data_processor
import scraper
import pandas as pd
import time
import json
import os
import signal
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global variables for graceful shutdown
running = True
session_folder = None
VERSION = '1.1' # Updated version for pure extraction

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
    csv_file = os.path.join(session_folder, "processed_companies.csv")
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        print(f"Found existing results: {len(existing_df)} companies already processed")
        return existing_df
    return None

def save_progress_realtime(session_folder, results, detailed_results, timestamp, current_company=""):
    """Save current progress in real-time"""
    # Save basic results to CSV
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(session_folder, "processed_companies.csv")
    results_df.to_csv(csv_file, index=False)
    
    # Save detailed results to JSON
    json_file = os.path.join(session_folder, "extracted_data.json")
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

def append_to_raw_text_file(session_folder, result):
    """Append a single result to the raw text file"""
    text_file = os.path.join(session_folder, "extracted_information.txt")
    with open(text_file, 'a', encoding='utf-8') as f:
        f.write(result['extracted_text'])
        f.write("\n\n")

def process_company(row, session_folder):
    import time
    company_start = time.time()
    company_name = row.company_name
    website_url = row.website
    domain_name = row.domain_name
    employee_count = getattr(row, 'no_of_employees', 'N/A')

    # Scrape website for raw text
    extracted_text = scraper.extract_website_text(website_url)
    
    status = "Success"
    if "Could not access website" in extracted_text:
        status = "Failed"

    company_time = time.time() - company_start
    result = {
        "Company Name": company_name,
        "Domain": domain_name,
        "Website": website_url,
        "Employee Count": employee_count,
        "Status": status,
        "detailed": {
            "company_name": company_name,
            "domain_name": domain_name,
            "website_url": website_url,
            "employee_count": employee_count,
            "status": status,
            "extracted_text": extracted_text,
            "extraction_timestamp": datetime.now().isoformat()
        },
        "company_time": company_time
    }
    # This append will now happen in order because the main loop processes sequentially
    append_to_raw_text_file(session_folder, result['detailed'])
    return result

def main():
    """
    Main function to run the sales intelligence bot for pure data extraction.
    """
    global running, session_folder
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command-line arguments for custom range
    parser = argparse.ArgumentParser(description="Sales Intelligence Bot - Data Extractor")
    parser.add_argument('--start', type=int, default=0, help='Start index (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
    parser.add_argument('--version', type=str, default=VERSION, help='Session version string')
    args = parser.parse_args()
    
    # Create session folder with range and version
    session_folder, timestamp = create_session_folder(args.start, args.end, args.version)
    print(f"Session folder: {session_folder}")
    print(f"📁 Results will be saved in real-time to: {session_folder}")
    print(f"⏹️  Press Ctrl+C to stop and save progress")
    
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
        companies_to_process = companies_df
        results = []
        print(f"🚀 Starting fresh: processing {len(companies_to_process)} companies")

    detailed_results = []
    
    print(f"\n{'='*60}")
    print("Starting data extraction... (Press Ctrl+C to stop anytime)")
    print(f"{'='*60}")
    
    max_workers = 10
    start_time = time.time()
    company_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks and keep them in a list to preserve order
        tasks = list(companies_to_process.itertuples(index=False))
        futures = [executor.submit(process_company, row, session_folder) for row in tasks]
        future_map = {future: tasks[i] for i, future in enumerate(futures)}

        # Iterate through the futures in the order they were submitted to process results sequentially
        for future in futures:
            if not running:
                # On shutdown, cancel all futures that haven't started
                for f in futures:
                    if not f.done():
                        f.cancel()
                break
            try:
                result = future.result()
                results.append({k: v for k, v in result.items() if k != 'detailed' and k != 'company_time'})
                detailed_results.append(result['detailed'])
                company_count += 1
                
                # Print summary for this company
                print(f"\n{'='*60}")
                print(f"[{company_count}/{len(companies_to_process)}] {result['Company Name']} ({result['Website']})")
                print(f"  Status: {result['Status']} | Time: {result['company_time']:.1f}s")
                print(f"{'='*60}")
                
                # Save progress in real-time. This will now be in order.
                save_progress_realtime(session_folder, results, detailed_results, timestamp, result['Company Name'])
            except Exception as e:
                company_row = future_map[future]
                print(f"An error occurred while processing {getattr(company_row, 'company_name', 'Unknown')}: {e}")

    total_time = time.time() - start_time
    
    # Final save
    if results:
        save_progress_realtime(session_folder, results, detailed_results, timestamp, "COMPLETED")
        print(f"\n{'='*60}")
        print(f"📊 Extraction Summary")
        print(f"{'='*60}")
        print(f"📁 Session folder: {session_folder}")
        print(f"📄 Files saved:")
        print(f"  - processed_companies.csv (list of companies and status)")
        print(f"  - extracted_data.json (full data with text)")
        print(f"  - extracted_information.txt (raw text output)")
        print(f"🕒 Total time: {total_time:.2f} seconds for {len(results)} companies")
    else:
        print("No companies were processed in this session.")

if __name__ == '__main__':
    main()

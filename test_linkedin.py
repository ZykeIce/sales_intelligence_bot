import requests
from bs4 import BeautifulSoup
import re
import urllib.parse

def extract_linkedin_company_info(html_content: str, linkedin_url: str):
    """Extract detailed company information from LinkedIn page"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        page_text = soup.get_text()
        
        # Check if it's a real company page
        if "Page not found" in page_text or "This page doesn't exist" in page_text:
            return "LinkedIn page not found"
        
        if "Sign in" in page_text and len(page_text) < 1000:
            return "LinkedIn requires authentication"
        
        # Extract company information
        info_parts = []
        
        # Company name and title
        title = soup.find('title')
        if title:
            title_text = title.get_text().strip()
            if "| LinkedIn" in title_text:
                company_name = title_text.split("| LinkedIn")[0].strip()
                info_parts.append(f"Company: {company_name}")
        
        # Meta description (often contains company description and follower count)
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            desc = meta_desc.get('content', '')
            # Extract follower count
            follower_match = re.search(r'(\d+(?:,\d+)*)\s+followers?', desc)
            if follower_match:
                info_parts.append(f"Followers: {follower_match.group(1)}")
            
            # Extract company description
            desc_clean = re.sub(r'[^a-zA-Z0-9\s,.-]', '', desc)
            if len(desc_clean) > 20:
                info_parts.append(f"Description: {desc_clean[:100]}...")
        
        # Look for job-related content
        job_indicators = []
        job_keywords = ['jobs', 'careers', 'hiring', 'open positions', 'apply', 'recruitment']
        for keyword in job_keywords:
            if re.search(rf'\b{keyword}\b', page_text, re.I):
                job_indicators.append(keyword)
        
        if job_indicators:
            info_parts.append(f"Job indicators found: {', '.join(job_indicators)}")
        
        # Look for company size indicators
        size_patterns = [
            r'(\d+(?:,\d+)*)\s+employees?',
            r'(\d+)\+?\s+employees?',
            r'company\s+size[:\s]*(\d+(?:,\d+)*)',
            r'(\d+(?:,\d+)*)\s+people'
        ]
        
        for pattern in size_patterns:
            size_match = re.search(pattern, page_text, re.I)
            if size_match:
                info_parts.append(f"Company size: {size_match.group(1)} employees")
                break
        
        # Look for industry information
        industry_patterns = [
            r'industry[:\s]*([^,\n]+)',
            r'sector[:\s]*([^,\n]+)',
            r'technology|software|healthcare|finance|retail|manufacturing'
        ]
        
        for pattern in industry_patterns:
            industry_match = re.search(pattern, page_text, re.I)
            if industry_match:
                info_parts.append(f"Industry: {industry_match.group(1).strip()}")
                break
        
        # Check for recent activity (posts, updates)
        if re.search(r'posts?|updates?|recent', page_text, re.I):
            info_parts.append("Recent activity detected")
        
        # Final assessment
        if len(info_parts) > 0:
            return f"LinkedIn page found: {linkedin_url} | " + " | ".join(info_parts)
        else:
            return f"LinkedIn page found: {linkedin_url} | Basic company page accessible"
            
    except Exception as e:
        return f"LinkedIn page found but error extracting info: {e}"

def test_linkedin_access():
    """Test if we can access LinkedIn company pages and extract information"""
    
    # Test companies from your data
    test_companies = [
        "Starkey Hearing",
        "Google", 
        "Microsoft",
        "Apple"
    ]
    
    print("Testing Enhanced LinkedIn Company Page Access")
    print("=" * 60)
    
    for company in test_companies:
        print(f"\nTesting: {company}")
        
        # Clean company name for LinkedIn URL
        clean_name = re.sub(r'[^\w\s-]', '', company).strip()
        linkedin_url = f"https://www.linkedin.com/company/{clean_name.lower().replace(' ', '-')}"
        
        print(f"LinkedIn URL: {linkedin_url}")
        
        try:
            # Try to access LinkedIn page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(linkedin_url, timeout=10, headers=headers)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                # Extract detailed information
                detailed_info = extract_linkedin_company_info(response.content, linkedin_url)
                print(f"Extracted Info: {detailed_info}")
                
            else:
                print(f"✗ Failed to access LinkedIn page (Status: {response.status_code})")
                
        except Exception as e:
            print(f"✗ Error accessing LinkedIn: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_linkedin_access() 
import requests
from bs4 import BeautifulSoup
import re
from openai import OpenAI
import config
import time
import urllib.parse
import json
from urllib.parse import urljoin

client = OpenAI(api_key=config.OPENAI_API_KEY)

COMMON_JOB_PATHS = [
    '/careers', '/jobs', '/join-us', '/work-with-us', '/about/careers', '/about-us/careers', '/about/jobs', '/company/careers', '/company/jobs'
]
COMMON_JOB_SUBDOMAINS = [
    'careers', 'jobs'
]

def scrape_website(url: str):
    """
    Scrapes a website and navigates through multiple pages to gather comprehensive information.

    Args:
        url: The URL of the website to scrape.

    Returns:
        A tuple containing:
        - is_hiring (bool): True if AI determines company is hiring.
        - hiring_reasoning (str): AI's detailed reasoning for hiring decision.
        - extracted_text (str): The text content from all explored pages.
        - additional_info (str): Information from additional sources.
    """
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return False, "Could not access website", None, ""

    soup = BeautifulSoup(response.content, 'lxml')

    # Navigate and gather comprehensive information
    all_text, navigation_info = navigate_website(url, soup)
    
    # Search for additional information
    additional_info = search_additional_sources(url, soup)
    
    # Combine all information
    combined_info = f"{additional_info} | {navigation_info}" if additional_info else navigation_info
    
    # Use AI to determine hiring status with reasoning
    is_hiring, hiring_reasoning = determine_hiring_with_ai(all_text, url, combined_info)
    
    return is_hiring, hiring_reasoning, all_text, combined_info

def navigate_website(base_url: str, main_soup: BeautifulSoup):
    """
    Navigate through multiple pages of a website using AI-driven free exploration.
    
    Args:
        base_url: The main website URL
        main_soup: BeautifulSoup object of the main page
        
    Returns:
        tuple: (combined_text, navigation_info)
    """
    all_text = []
    navigation_info = []
    visited_urls = set()
    
    # Start with main page
    if main_soup.body:
        for script_or_style in main_soup(["script", "style"]):
            script_or_style.decompose()
        main_text = main_soup.body.get_text(separator=' ', strip=True)
        all_text.append(main_text)
        visited_urls.add(base_url)
    
    # Use AI to explore the website freely
    print("  Starting AI-driven free exploration...")
    explored_text, explored_info = ai_free_exploration(base_url, main_soup, visited_urls)
    
    all_text.extend(explored_text)
    navigation_info.extend(explored_info)
    
    # Combine all text (limit to avoid token limits)
    combined_text = " ".join(all_text)
    combined_text = " ".join(combined_text.split()[:3500])  # Limit to 3500 words
    
    return combined_text, " | ".join(navigation_info) if navigation_info else "AI exploration completed."

def ai_free_exploration(base_url: str, soup: BeautifulSoup, visited_urls: set, max_depth=3, max_pages=12):
    """
    Explore a website freely using AI to decide where to go next.
    
    Args:
        base_url: The main website URL
        soup: BeautifulSoup object of the current page
        visited_urls: Set of visited URLs
        max_depth: Maximum recursion depth to prevent getting lost
        max_pages: Maximum pages to explore to prevent runaway requests
        
    Returns:
        tuple: (list_of_texts, list_of_infos)
    """
    # Stop conditions
    if max_depth <= 0 or len(visited_urls) >= max_pages:
        return [], []

    # Extract links from the current page
    potential_links = extract_navigation_links(soup, base_url)
    if not potential_links:
        return [], []

    # Use AI to decide which link to follow next
    next_action, next_url, reasoning = ai_decide_next_action(soup.get_text(), potential_links, base_url)

    if next_action == 'STOP' or not next_url:
        return [], [f"AI decided to stop exploring: {reasoning}"]

    if next_url in visited_urls:
        return [], [f"AI wanted to visit an already explored page. Stopping this path."]

    try:
        # print(f"  AI decided to explore: {next_url}")
        # print(f"    Reasoning: {reasoning}")

        response = requests.get(next_url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            visited_urls.add(next_url)
            page_soup = BeautifulSoup(response.content, 'lxml')
            
            page_text = ""
            if page_soup.body:
                for script_or_style in page_soup(["script", "style"]):
                    script_or_style.decompose()
                page_text = page_soup.body.get_text(separator=' ', strip=True)

            page_analysis = f"Explored '{next_url}'."

            # Recursively explore from the new page
            sub_texts, sub_infos = ai_free_exploration(base_url, page_soup, visited_urls, max_depth - 1, max_pages)

            return [page_text] + sub_texts, [page_analysis] + sub_infos

        else:
            return [], [f"Could not access AI-selected page: {next_url}"]

    except Exception as e:
        print(f"    Error exploring {next_url}: {e}")
        return [], [f"Error accessing AI-selected page: {next_url}"]

def ai_decide_next_action(page_text: str, links: list, base_url: str):
    """
    Use AI to decide which link to follow next or if it's satisfied.
    """
    # Prepare link information for AI
    link_info = []
    for url, link_text, context in links[:20]: # Limit links sent to AI
        link_info.append(f"Link: '{link_text}' -> {url} (Context: {context})")

    links_text = "\n".join(link_info)

    prompt = f"""
    You are an intelligent web explorer trying to find hiring information about a company.
    Based on the current page content and available links, decide what to do next.
    
    Current page content summary: {page_text[:1000]}
    
    Available links:
    {links_text}
    
    YOUR TASK:
    1. Analyze the current page. Have you found strong hiring indicators (like job listings)?
    2. Analyze the links. Which one is most likely to lead to more hiring information?
    3. Decide your next action: EXPLORE or STOP.
    
    - If you have found strong evidence of hiring (like a list of open positions) or no more relevant links, choose STOP.
    - If there are promising links that could lead to hiring info (like 'Careers', 'About Us', 'Jobs'), choose EXPLORE and select the single best link.
    
    Respond in this exact format, with no extra text:
    
    ACTION: [EXPLORE/STOP]
    URL: [full_url_to_explore_or_NA]
    REASONING: [Your detailed explanation for why you chose this action, or why you decided to stop]
    """
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an intelligent web explorer. Your mission is to find hiring information efficiently by deciding which links to follow. Provide your response in the specified format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300,
        )
        result = response.choices[0].message.content.strip()
        
        action, url, reasoning = "STOP", "NA", "Could not parse AI decision."
        lines = result.split('\n')
        for line in lines:
            if "ACTION:" in line:
                action = line.split("ACTION:", 1)[1].strip()
            elif "URL:" in line:
                url = line.split("URL:", 1)[1].strip()
            elif "REASONING:" in line:
                reasoning = line.split("REASONING:", 1)[1].strip()
        
        if url == "NA": url = ""
        return action, url, reasoning
        
    except Exception as e:
        print(f"Error with AI action decision: {e}")
        return "STOP", "", "Error occurred during AI decision"

def extract_navigation_links(soup: BeautifulSoup, base_url: str):
    """
    Extract navigation links from HTML source code.
    
    Args:
        soup: BeautifulSoup object of the page
        base_url: The main website URL
        
    Returns:
        list: List of (url, link_text, context) tuples
    """
    relevant_links = []
    
    # Look for navigation elements
    nav_selectors = [
        'nav', 'header', '.navigation', '.nav', '.menu', '.navbar',
        '#navigation', '#nav', '#menu', '#navbar'
    ]
    
    navigation_elements = []
    for selector in nav_selectors:
        elements = soup.select(selector)
        navigation_elements.extend(elements)
    
    # If no specific nav elements found, look for common navigation patterns
    if not navigation_elements:
        # Look for links in header, footer, or main content areas
        navigation_elements = soup.find_all(['header', 'footer', 'main'])
    
    # Extract links from navigation elements
    for nav_element in navigation_elements:
        links = nav_element.find_all('a', href=True)
        for link in links:
            href = link.get('href', '')
            link_text = link.get_text().strip()
            
            # Skip empty or very short link text
            if len(link_text) < 2:
                continue
            
            # Make absolute URL
            if href.startswith('/'):
                full_url = base_url.rstrip('/') + href
            elif href.startswith('http'):
                full_url = href
            else:
                full_url = base_url.rstrip('/') + '/' + href
            
            # Only include if it's the same domain
            if base_url.split('/')[2] in full_url:
                # Get context (parent element text or nearby text)
                context = get_link_context(link)
                relevant_links.append((full_url, link_text, context))
    
    # If no navigation elements found, get all links from the page
    if not relevant_links:
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            link_text = link.get_text().strip()
            
            if len(link_text) > 2:
                # Make absolute URL
                if href.startswith('/'):
                    full_url = base_url.rstrip('/') + href
                elif href.startswith('http'):
                    full_url = href
                else:
                    full_url = base_url.rstrip('/') + '/' + href
                
                # Only include if it's the same domain
                if base_url.split('/')[2] in full_url:
                    context = get_link_context(link)
                    relevant_links.append((full_url, link_text, context))
    
    # Sort by relevance (prioritize links that seem most relevant to hiring)
    relevant_links.sort(key=lambda x: calculate_link_relevance(x[1], x[2]), reverse=True)
    
    return relevant_links

def get_link_context(link_element):
    """
    Get context around a link to understand what it's about.
    
    Args:
        link_element: BeautifulSoup link element
        
    Returns:
        str: Context information
    """
    context_parts = []
    
    # Get parent element text
    parent = link_element.parent
    if parent:
        parent_text = parent.get_text().strip()
        if parent_text and parent_text != link_element.get_text().strip():
            context_parts.append(f"Parent: {parent_text[:50]}...")
    
    # Get nearby text (siblings)
    siblings = link_element.find_previous_siblings()
    for sibling in siblings[:2]:  # Look at 2 previous siblings
        sibling_text = sibling.get_text().strip()
        if sibling_text and len(sibling_text) > 5:
            context_parts.append(f"Nearby: {sibling_text[:30]}...")
            break
    
    return " | ".join(context_parts) if context_parts else "No context"

def calculate_link_relevance(link_text: str, context: str):
    """
    Calculate how relevant a link is for hiring analysis.
    
    Args:
        link_text: The link text
        context: Context around the link
        
    Returns:
        int: Relevance score (higher = more relevant)
    """
    score = 0
    all_text = f"{link_text} {context}".lower()
    
    # High relevance keywords
    high_relevance = ['careers', 'jobs', 'hiring', 'employment', 'work with us', 'join our team']
    for keyword in high_relevance:
        if keyword in all_text:
            score += 10
    
    # Medium relevance keywords
    medium_relevance = ['about', 'team', 'company', 'people', 'culture']
    for keyword in medium_relevance:
        if keyword in all_text:
            score += 5
    
    # Low relevance keywords
    low_relevance = ['contact', 'news', 'blog', 'press']
    for keyword in low_relevance:
        if keyword in all_text:
            score += 2
    
    return score

def analyze_page_content(page_text: str, link_text: str, page_url: str):
    """
    Analyze the content of a page to see what hiring-related information it contains.
    
    Args:
        page_text: Text content of the page
        link_text: The link text that led to this page
        page_url: URL of the page
        
    Returns:
        str: Analysis of what was found
    """
    findings = []
    
    # Look for specific hiring indicators
    if re.search(r'open positions?|job listings?|current openings', page_text, re.I):
        findings.append("Found job listings")
    
    if re.search(r'apply now|apply online|submit application', page_text, re.I):
        findings.append("Found application process")
    
    if re.search(r'we\'re hiring|join our team|work with us', page_text, re.I):
        findings.append("Found hiring messaging")
    
    if re.search(r'careers?|jobs?|employment', page_text, re.I):
        findings.append("Found careers content")
    
    if re.search(r'growing|expanding|hiring|recruiting', page_text, re.I):
        findings.append("Found growth/hiring mentions")
    
    if re.search(r'team|people|culture', page_text, re.I):
        findings.append("Found team/company info")
    
    if re.search(r'contact.*careers?|hr.*contact|talent.*contact', page_text, re.I):
        findings.append("Found career contact info")
    
    # If no specific findings, provide general assessment
    if not findings:
        if len(page_text) > 500:
            findings.append("Page contains substantial content")
        else:
            findings.append("Page has limited content")
    
    return "; ".join(findings) if findings else "No specific hiring indicators found"

def ai_driven_page_selection(base_url: str, soup: BeautifulSoup, main_page_text: str):
    """
    Use AI to intelligently select which pages to explore based on website content.
    
    Args:
        base_url: The main website URL
        soup: BeautifulSoup object of the main page
        main_page_text: Text content of the main page
        
    Returns:
        list: List of (url, page_type, ai_reasoning) tuples
    """
    # Extract all links from the page
    links = soup.find_all('a', href=True)
    potential_pages = []
    
    for link in links:
        href = link.get('href', '')
        link_text = link.get_text().strip()
        
        # Make absolute URL
        if href.startswith('/'):
            full_url = base_url.rstrip('/') + href
        elif href.startswith('http'):
            full_url = href
        else:
            full_url = base_url.rstrip('/') + '/' + href
        
        # Only include if it's the same domain
        if base_url.split('/')[2] in full_url and len(link_text) > 0:
            potential_pages.append((full_url, link_text))
    
    if not potential_pages:
        return []
    
    # Use AI to analyze the main page and select relevant pages
    return ai_select_relevant_pages(main_page_text, potential_pages, base_url)

def ai_select_relevant_pages(main_page_text: str, potential_pages: list, base_url: str):
    """
    Use AI to intelligently select which pages are most relevant for hiring analysis.
    
    Args:
        main_page_text: Text content of the main page
        potential_pages: List of (url, link_text) tuples
        base_url: The main website URL
        
    Returns:
        list: List of (url, page_type, ai_reasoning) tuples
    """
    # Prepare the list of pages for AI analysis
    pages_info = []
    for url, link_text in potential_pages[:20]:  # Limit to first 20 to avoid token limits
        pages_info.append(f"Link: '{link_text}' -> {url}")
    
    pages_text = "\n".join(pages_info)
    
    prompt = f"""
    Analyze this company's main page content and available navigation links to determine which pages would be most relevant for understanding if they are hiring.
    
    Main page content: {main_page_text[:1000]}
    
    Available navigation links:
    {pages_text}
    
    Based on the main page content and available links, select the 3-5 most relevant pages for hiring analysis. Consider:
    1. Pages that might contain job listings or careers information
    2. Pages that might mention company growth, expansion, or hiring
    3. Pages that might contain team or company information
    4. Pages that might have contact information for job applications
    
    Respond in this exact format:
    
    SELECTED_PAGES:
    1. URL: [full_url] | TYPE: [page_type] | REASONING: [why this page is relevant]
    2. URL: [full_url] | TYPE: [page_type] | REASONING: [why this page is relevant]
    ...
    
    Page types can be: Careers, About, Team, Contact, News, Culture, or other relevant categories.
    """
    
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a website navigation expert. Analyze website content and navigation to identify the most relevant pages for hiring analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the AI response
        selected_pages = []
        lines = result.split('\n')
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                # Parse: "1. URL: [url] | TYPE: [type] | REASONING: [reasoning]"
                parts = line.split('|')
                if len(parts) >= 3:
                    url_part = parts[0].split('URL:')[1].strip() if 'URL:' in parts[0] else ''
                    type_part = parts[1].split('TYPE:')[1].strip() if 'TYPE:' in parts[1] else 'Unknown'
                    reasoning_part = parts[2].split('REASONING:')[1].strip() if 'REASONING:' in parts[2] else ''
                    
                    if url_part:
                        selected_pages.append((url_part, type_part, reasoning_part))
        
        return selected_pages
        
    except Exception as e:
        print(f"Error with AI page selection: {e}")
        # Fallback to basic selection
        return basic_page_selection(potential_pages)

def basic_page_selection(potential_pages: list):
    """Fallback method for page selection"""
    selected = []
    for url, link_text in potential_pages[:3]:
        if any(keyword in link_text.lower() for keyword in ['careers', 'jobs', 'about', 'team', 'contact']):
            selected.append((url, 'Basic Selection', f'Found keyword in link: {link_text}'))
    return selected

def find_relevant_pages(base_url: str, soup: BeautifulSoup):
    """
    Find relevant pages to explore on the website.
    
    Args:
        base_url: The main website URL
        soup: BeautifulSoup object of the main page
        
    Returns:
        list: List of (url, page_type) tuples
    """
    relevant_pages = []
    
    # Define relevant page patterns
    page_patterns = [
        (r'careers?|jobs?|hiring|employment|work-with-us|join-our-team', 'Careers/Jobs Page'),
        (r'about|about-us|company|our-story', 'About Page'),
        (r'team|leadership|people', 'Team Page'),
        (r'contact|contact-us|get-in-touch', 'Contact Page'),
        (r'news|blog|press|media', 'News/Blog Page'),
        (r'culture|values|mission', 'Culture Page')
    ]
    
    # Find all links
    links = soup.find_all('a', href=True)
    
    for link in links:
        href = link.get('href', '').lower()
        link_text = link.get_text().lower()
        
        # Check if link matches any relevant pattern
        for pattern, page_type in page_patterns:
            if re.search(pattern, href) or re.search(pattern, link_text):
                # Make absolute URL
                if href.startswith('/'):
                    full_url = base_url.rstrip('/') + href
                elif href.startswith('http'):
                    full_url = href
                else:
                    full_url = base_url.rstrip('/') + '/' + href
                
                # Only include if it's the same domain
                if base_url.split('/')[2] in full_url:
                    relevant_pages.append((full_url, page_type))
                    break
    
    return relevant_pages

def check_page_for_hiring_indicators(page_text: str, page_type: str):
    """
    Check a specific page for hiring indicators.
    
    Args:
        page_text: Text content of the page
        page_type: Type of page (Careers, About, etc.)
        
    Returns:
        str: Hiring indicators found on this page
    """
    indicators = []
    
    # Different indicators for different page types
    if page_type == 'Careers/Jobs Page':
        if re.search(r'open positions?|job listings?|apply now|current openings', page_text, re.I):
            indicators.append("Active job listings found")
        if re.search(r'careers?|jobs?|hiring', page_text, re.I):
            indicators.append("Careers content found")
        if re.search(r'apply|application|submit resume', page_text, re.I):
            indicators.append("Application process found")
    
    elif page_type == 'About Page':
        if re.search(r'growing|expanding|hiring|recruiting', page_text, re.I):
            indicators.append("Growth/hiring mentioned")
        if re.search(r'team|join us|work with us', page_text, re.I):
            indicators.append("Team/joining content found")
    
    elif page_type == 'Team Page':
        if re.search(r'join our team|we\'re hiring|open positions', page_text, re.I):
            indicators.append("Hiring messaging found")
        if re.search(r'careers?|jobs?', page_text, re.I):
            indicators.append("Job-related content found")
    
    elif page_type == 'Contact Page':
        if re.search(r'careers?|jobs?|employment|recruitment', page_text, re.I):
            indicators.append("Career contact info found")
        if re.search(r'hr|human resources|talent', page_text, re.I):
            indicators.append("HR contact found")
    
    elif page_type == 'News/Blog Page':
        if re.search(r'hiring|recruiting|new team members|job openings', page_text, re.I):
            indicators.append("Hiring news found")
        if re.search(r'growth|expansion|new positions', page_text, re.I):
            indicators.append("Growth announcements found")
    
    return "; ".join(indicators) if indicators else "No specific hiring indicators"

def search_additional_sources(base_url: str, soup: BeautifulSoup):
    """
    Search for additional information from careers pages, LinkedIn, and other sources.
    
    Args:
        base_url: The main website URL
        soup: BeautifulSoup object of the main page
        
    Returns:
        str: Additional information found
    """
    additional_info = []
    
    # Extract domain for LinkedIn search
    domain = extract_domain(base_url)
    company_name = extract_company_name(soup)
    
    # 1. Look for careers/jobs pages on the main site
    careers_info = search_careers_pages(base_url, soup)
    if careers_info:
        additional_info.append(f"Careers page info: {careers_info}")
    
    # 2. Try to find LinkedIn company page
    linkedin_info = search_linkedin_company(company_name, domain)
    if linkedin_info:
        additional_info.append(f"LinkedIn info: {linkedin_info}")
    
    # 3. Search for job boards or recruitment pages
    job_boards_info = search_job_boards(base_url, soup)
    if job_boards_info:
        additional_info.append(f"Job boards info: {job_boards_info}")
    
    return " | ".join(additional_info) if additional_info else "No additional sources found"

def extract_domain(url: str):
    """Extract domain from URL"""
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc.replace('www.', '')
    except:
        return url.replace('http://', '').replace('https://', '').replace('www.', '')

def extract_company_name(soup: BeautifulSoup):
    """Extract company name from page title or meta tags"""
    # Try title first
    title = soup.find('title')
    if title:
        title_text = title.get_text().strip()
        # Clean up common title suffixes
        for suffix in [' - Home', ' | Home', ' - Welcome', ' | Welcome', ' - Official Site']:
            title_text = title_text.replace(suffix, '')
        return title_text
    
    # Try meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        return meta_desc.get('content', '')[:50]
    
    return ""

def search_careers_pages(base_url: str, soup: BeautifulSoup):
    """Search for careers/jobs pages on the main site"""
    careers_info = []
    
    # Look for careers/jobs links
    career_links = soup.find_all('a', href=re.compile(r'careers?|jobs?|hiring|employment', re.I))
    
    for link in career_links[:3]:  # Limit to first 3
        href = link.get('href', '')
        if href:
            # Make absolute URL
            if href.startswith('/'):
                full_url = base_url.rstrip('/') + href
            elif href.startswith('http'):
                full_url = href
            else:
                full_url = base_url.rstrip('/') + '/' + href
            
            try:
                print(f"  Checking careers page: {full_url}")
                response = requests.get(full_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    careers_soup = BeautifulSoup(response.content, 'lxml')
                    careers_text = careers_soup.get_text()[:500]  # First 500 chars
                    if re.search(r'open positions?|job listings?|apply now|current openings', careers_text, re.I):
                        careers_info.append(f"Active careers page found with job listings")
                    else:
                        careers_info.append(f"Careers page exists but no clear job listings")
                time.sleep(1)  # Be respectful
            except:
                careers_info.append(f"Careers page found but could not access")
    
    return " | ".join(careers_info) if careers_info else ""

def search_linkedin_company(company_name: str, domain: str):
    """Search for LinkedIn company page and extract detailed information"""
    if not company_name and not domain:
        return ""
    
    # Try to construct LinkedIn URL
    search_terms = [company_name, domain]
    
    for term in search_terms:
        if not term:
            continue
            
        # Clean the term
        clean_term = re.sub(r'[^\w\s-]', '', term).strip()
        if len(clean_term) < 3:
            continue
            
        # Try LinkedIn company URL pattern
        linkedin_url = f"https://www.linkedin.com/company/{clean_term.lower().replace(' ', '-')}"
        
        try:
            print(f"  Checking LinkedIn: {linkedin_url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(linkedin_url, timeout=10, headers=headers)
            if response.status_code == 200:
                linkedin_info = extract_linkedin_company_info(response.content, linkedin_url)
                return linkedin_info
            time.sleep(1)
        except Exception as e:
            print(f"    LinkedIn access error: {e}")
            continue
    
    return ""

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

def search_job_boards(base_url: str, soup: BeautifulSoup):
    """Search for job board integrations or links"""
    job_boards = []
    
    # Look for common job board integrations
    job_board_patterns = [
        r'lever\.co', r'greenhouse\.io', r'workday\.com', r'bamboohr\.com',
        r'jobvite\.com', r'icims\.com', r'workable\.com', r'ziprecruiter\.com'
    ]
    
    page_text = soup.get_text()
    for pattern in job_board_patterns:
        if re.search(pattern, page_text, re.I):
            job_boards.append(f"Job board integration found: {pattern}")
    
    return " | ".join(job_boards) if job_boards else ""

def try_common_job_sections(base_url, visited_urls):
    """Try common job/careers subdomains and paths if not found in navigation."""
    found_urls = []
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    domain = parsed.netloc.replace('www.', '')
    scheme = parsed.scheme
    # Try subdomains
    for sub in COMMON_JOB_SUBDOMAINS:
        job_url = f"{scheme}://{sub}.{domain}"
        if job_url not in visited_urls:
            try:
                resp = requests.get(job_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200 and len(resp.content) > 500:
                    found_urls.append(job_url)
                    visited_urls.add(job_url)
            except: pass
    # Try common paths
    for path in COMMON_JOB_PATHS:
        job_url = urljoin(base_url, path)
        if job_url not in visited_urls:
            try:
                resp = requests.get(job_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200 and len(resp.content) > 500:
                    found_urls.append(job_url)
                    visited_urls.add(job_url)
            except: pass
    return found_urls

def parse_schema_org_jobposting(html):
    """Parse schema.org JobPosting markup from HTML."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    jobs = []
    for tag in soup.find_all(attrs={"itemtype": re.compile("JobPosting", re.I)}):
        job = {}
        for prop in tag.find_all(attrs={"itemprop": True}):
            job[prop['itemprop']] = prop.get_text(strip=True)
        if job:
            jobs.append(job)
    # Also check for JSON-LD
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get('@type') == 'JobPosting':
                jobs.append(data)
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and entry.get('@type') == 'JobPosting':
                        jobs.append(entry)
        except: pass
    return jobs

def determine_hiring_with_ai(website_text: str, url: str, additional_info: str):
    """
    Uses AI to intelligently determine if a company is hiring with detailed reasoning.
    
    Args:
        website_text: Text extracted from the company website
        url: The website URL
        additional_info: Information from additional sources
        
    Returns:
        tuple: (is_hiring: bool, reasoning: str)
    """
    if not website_text and not additional_info:
        return False, "No website content could be extracted"
    
    prompt = f"""
    Analyze this company's website content and additional sources to determine if they are actively hiring.
    Website: {url}
    Website content: {website_text}
    Additional sources info: {additional_info}
    
    Look for CONCRETE EVIDENCE of job availability:
    - Only mark DECISION: YES if you find actual job listings, open positions, job application forms, or explicit evidence that jobs are available and can be applied for now.
    - The presence of a careers page, jobs page, or generic hiring messaging is NOT enough unless it contains specific, current job openings or application instructions.
    - If you find schema.org JobPosting markup or job board integrations with open roles, this counts as concrete evidence.
    - If there is no explicit evidence of available jobs, mark DECISION: NO.
    
    Provide your analysis in this exact format:
    DECISION: [YES/NO]
    REASONING: [Your detailed explanation, including a list of all job listings, job board links, and application forms found. Be thorough.]
    """
    
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a hiring detection expert. Only mark YES if there is concrete evidence of job availability (job listings, open positions, or application forms). Be strict and require explicit evidence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=400,
        )
        result = response.choices[0].message.content.strip()
        if "DECISION: YES" in result.upper():
            is_hiring = True
        elif "DECISION: NO" in result.upper():
            is_hiring = False
        else:
            # Fallback: Only set YES if explicit job evidence keywords are present
            if any(keyword in result.upper() for keyword in ["JOB LISTING", "OPEN POSITION", "APPLY NOW", "APPLICATION FORM", "CURRENT OPENINGS"]):
                is_hiring = True
            else:
                is_hiring = False
        reasoning_start = result.find("REASONING:")
        if reasoning_start != -1:
            reasoning = result[reasoning_start + 10:].strip()
        else:
            reasoning = "AI provided analysis but reasoning format was unclear"
        return is_hiring, reasoning
    except Exception as e:
        print(f"Error with AI hiring detection: {e}")
        all_text = f"{website_text} {additional_info}"
        # Fallback: Only set YES if explicit job evidence keywords are present
        hiring_keywords = ['job listing', 'open position', 'apply now', 'application form', 'current openings']
        found_keywords = []
        for keyword in hiring_keywords:
            if re.search(rf'\b{keyword}\b', all_text, re.I):
                found_keywords.append(keyword)
        if found_keywords:
            return True, f"Fallback: Found explicit job evidence: {', '.join(found_keywords)}"
        else:
            return False, "Fallback: No explicit job evidence found across sources"

# --- ENHANCED CAREERS PAGE EXPLORATION ---
def explore_careers_page_and_follow_links(careers_url, visited_urls, max_depth=2):
    """Fetch the careers page, extract and follow relevant links (job boards, application forms, job subpages)."""
    from bs4 import BeautifulSoup
    import re
    results = []
    try:
        resp = requests.get(careers_url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        if resp.status_code == 200:
            html = resp.content
            soup = BeautifulSoup(html, 'lxml')
            page_text = soup.get_text(" ", strip=True)
            results.append(page_text)
            # Find relevant links (job boards, application forms, job subpages)
            for a in soup.find_all('a', href=True):
                href = a['href']
                link_text = a.get_text().lower()
                # Heuristics: follow if link looks like job board, application, or job listing
                if any(kw in href for kw in ['lever.co', 'greenhouse.io', 'workday.com', 'bamboohr.com', 'jobvite.com', 'icims.com', 'workable.com', 'ziprecruiter.com']) or \
                   any(kw in link_text for kw in ['apply', 'open positions', 'job', 'opening', 'position', 'opportunities']):
                    # Make absolute URL
                    if href.startswith('http'):
                        next_url = href
                    else:
                        from urllib.parse import urljoin
                        next_url = urljoin(careers_url, href)
                    if next_url not in visited_urls and max_depth > 0:
                        visited_urls.add(next_url)
                        try:
                            sub_resp = requests.get(next_url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
                            if sub_resp.status_code == 200:
                                sub_soup = BeautifulSoup(sub_resp.content, 'lxml')
                                sub_text = sub_soup.get_text(" ", strip=True)
                                results.append(sub_text)
                        except: pass
    except: pass
    return results

# --- PATCH scrape_website to use enhanced careers page exploration ---
old_scrape_website = scrape_website

def scrape_website(url: str):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return False, "Could not access website", None, ""
    soup = BeautifulSoup(response.content, 'lxml')
    all_text, navigation_info = navigate_website(url, soup)
    visited_urls = set([url])
    # Try common job/careers subdomains and paths
    job_section_urls = try_common_job_sections(url, visited_urls)
    job_section_texts = []
    for job_url in job_section_urls:
        # Enhanced: Explore careers page and follow relevant links
        job_section_texts.extend(explore_careers_page_and_follow_links(job_url, visited_urls, max_depth=2))
    # Parse schema.org JobPosting from all HTML
    job_postings = parse_schema_org_jobposting(response.content)
    for job_url in job_section_urls:
        try:
            resp = requests.get(job_url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code == 200:
                job_postings.extend(parse_schema_org_jobposting(resp.content))
        except: pass
    # Search for additional info (job boards, LinkedIn, etc.)
    additional_info = search_additional_sources(url, soup)
    # Combine all info
    combined_info = f"{additional_info} | {navigation_info}"
    if job_section_urls:
        combined_info += f" | Job section URLs: {', '.join(job_section_urls)}"
    if job_postings:
        combined_info += f" | Job postings found: {len(job_postings)}"
    # Use AI to determine hiring status with reasoning
    is_hiring, hiring_reasoning = determine_hiring_with_ai(
        all_text + ' ' + ' '.join(job_section_texts), url, combined_info
    )
    return is_hiring, hiring_reasoning, all_text + ' ' + ' '.join(job_section_texts), combined_info

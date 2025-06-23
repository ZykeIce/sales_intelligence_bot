import requests
from bs4 import BeautifulSoup
import re
from openai import OpenAI
import config
import time
import urllib.parse
from urllib.parse import urljoin, urlparse, urlunparse

client = OpenAI(api_key=config.OPENAI_API_KEY)


def normalize_url(url: str):
    """Normalizes a URL for consistent comparison by converting to https, removing www., and stripping trailing slashes/queries."""
    try:
        url = url.strip()
        parts = urlparse(url)
        
        scheme = parts.scheme
        if scheme == 'http':
            scheme = 'https'
        
        netloc = parts.netloc
        if netloc.startswith('www.'):
            netloc = netloc[4:]
            
        path = parts.path.rstrip('/')
        
        # Reconstruct the URL without query params or fragments
        # An empty path is valid for the domain root.
        normalized_parts = (scheme, netloc, path, '', '', '')
        return urlunparse(normalized_parts)
    except Exception:
        # If parsing fails for any reason, return the original URL stripped of whitespace
        return url.strip()

def extract_website_text(url: str):
    """
    Scrapes a website to extract its text content using AI-driven navigation.

    Args:
        url: The URL of the website to scrape.

    Returns:
        A string containing all the extracted text from the website.
    """
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        # After the initial request, use the final URL from the response.
        # This handles redirects (e.g., http to https) correctly.
        final_url = response.url
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return f"Could not access website: {e}"

    soup = BeautifulSoup(response.content, 'lxml')

    # Navigate and gather comprehensive information, using the final URL as the base
    all_text, navigation_info = navigate_website(final_url, soup)
    
    # Search for additional information from LinkedIn
    additional_info = search_linkedin(final_url, soup)
    
    # Combine all information into a single text block
    full_extracted_text = f"--- START OF WEBSITE: {final_url} ---\n\n"
    full_extracted_text += f"INITIAL PAGE CONTENT:\n{all_text}\n\n"
    full_extracted_text += f"NAVIGATION INFO: {navigation_info}\n\n"
    full_extracted_text += f"ADDITIONAL INFO: {additional_info}\n\n"
    full_extracted_text += f"--- END OF WEBSITE: {final_url} ---\n"
    
    return full_extracted_text

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
    
    # Start with main page, using a normalized URL for tracking
    normalized_base_url = normalize_url(base_url)
    visited_urls.add(normalized_base_url)

    if main_soup.body:
        for script_or_style in main_soup(["script", "style"]):
            script_or_style.decompose()
        main_text = main_soup.body.get_text(separator=' ', strip=True)
        all_text.append(main_text)
    
    # Use AI to explore the website freely
    print("  Starting AI-driven free exploration...")
    explored_text, explored_info = ai_free_exploration(base_url, main_soup, visited_urls)
    
    all_text.extend(explored_text)
    navigation_info.extend(explored_info)
    
    # Combine all text (limit to avoid token limits)
    combined_text = " ".join(all_text)
    combined_text = " ".join(combined_text.split()[:4000])  # Limit to 4000 words
    
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
    next_action, next_url, reasoning = ai_decide_next_action(soup.get_text(), potential_links, base_url, visited_urls)

    if next_action == 'STOP' or not next_url:
        return [], [f"AI decided to stop exploring: {reasoning}"]

    normalized_next_url = normalize_url(next_url)
    if normalized_next_url in visited_urls:
        return [], [f"AI wanted to visit an already explored page ({next_url}). Stopping this path."]

    try:
        response = requests.get(next_url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            visited_urls.add(normalized_next_url)
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

def ai_decide_next_action(page_text: str, links: list, base_url: str, visited_urls: set):
    """
    Use AI to decide which link to follow next.
    """
    # Prepare link information for AI, excluding already visited links (using normalization)
    link_info = []
    unvisited_links = []
    for url, link_text, context in links:
        if normalize_url(url) not in visited_urls:
            unvisited_links.append((url, link_text, context))

    if not unvisited_links:
        return "STOP", "", "No unvisited links found to explore."

    for url, link_text, context in unvisited_links[:20]: # Limit links sent to AI
        link_info.append(f"Link: '{link_text}' -> {url} (Context: {context})")

    links_text = "\n".join(link_info)
    
    # We no longer need to pass the visited list in the prompt, as we pre-filter the links
    prompt = f"""
    You are an intelligent web explorer trying to gather information about a company.
    Your goal is to understand what the company does, its products/services, and who they are by visiting relevant pages.
    Based on the current page content and available links, decide what to do next.
    
    Current page content summary: {page_text[:1000]}
    
    Available UNVISITED links to explore:
    {links_text}
    
    YOUR TASK:
    1. Analyze the links. Which one is most likely to lead to key company information (About Us, Products, Services, Team, News, Technology)?
    2. Decide your next action: EXPLORE or STOP.
    
    - If there are promising links that could lead to more info, choose EXPLORE and select the single best UNVISITED link.
    - If you feel you have explored enough or there are no more relevant links to explore, choose STOP.
    
    Respond in this exact format, with no extra text:
    
    ACTION: [EXPLORE/STOP]
    URL: [full_url_to_explore_or_NA]
    REASONING: [Your detailed explanation for why you chose this action]
    """
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an intelligent web explorer. Your mission is to find key company information efficiently by deciding which links to follow. Provide your response in the specified format."},
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
            full_url = urljoin(base_url, href)
            
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
                full_url = urljoin(base_url, href)
                
                # Only include if it's the same domain
                if base_url.split('/')[2] in full_url:
                    context = get_link_context(link)
                    relevant_links.append((full_url, link_text, context))
    
    # Sort by relevance
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
    Calculate how relevant a link is for general company information analysis.
    
    Args:
        link_text: The link text
        context: Context around the link
        
    Returns:
        int: Relevance score (higher = more relevant)
    """
    score = 0
    all_text = f"{link_text} {context}".lower()
    
    # High relevance keywords
    high_relevance = ['about', 'company', 'what we do', 'services', 'products', 'solutions', 'platform', 'technology']
    for keyword in high_relevance:
        if keyword in all_text:
            score += 10
    
    # Medium relevance keywords
    medium_relevance = ['team', 'people', 'culture', 'news', 'blog', 'press', 'investors']
    for keyword in medium_relevance:
        if keyword in all_text:
            score += 5
    
    # Low relevance keywords
    low_relevance = ['contact', 'support', 'login', 'demo', 'careers', 'jobs']
    for keyword in low_relevance:
        if keyword in all_text:
            score += 2
    
    # Penalize irrelevant links
    irrelevant_keywords = ['privacy policy', 'terms of service', 'sitemap', 'accessibility']
    for keyword in irrelevant_keywords:
        if keyword in all_text:
            score -= 10
            
    return score

def search_linkedin(base_url: str, soup: BeautifulSoup):
    """
    Search for LinkedIn company page.
    
    Args:
        base_url: The main website URL
        soup: BeautifulSoup object of the main page
        
    Returns:
        str: LinkedIn information found
    """
    # Extract domain for LinkedIn search
    domain = extract_domain(base_url)
    company_name = extract_company_name(soup)
    
    linkedin_info = search_linkedin_company(company_name, domain)
    if linkedin_info:
        return f"LinkedIn info: {linkedin_info}"
    
    return "No additional sources found"

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
        
        # Look for company size indicators
        size_patterns = [r'(\d+(?:,\d+)*)\s+employees?', r'(\d+)\+?\s+employees?', r'company\s+size[:\s]*(\d+(?:,\d+)*)', r'(\d+(?:,\d+)*)\s+people']
        for pattern in size_patterns:
            size_match = re.search(pattern, page_text, re.I)
            if size_match:
                info_parts.append(f"Company size: {size_match.group(1)} employees")
                break
        
        # Look for industry information
        industry_patterns = [r'industry[:\s]*([^,\n]+)', r'sector[:\s]*([^,\n]+)', r'technology|software|healthcare|finance|retail|manufacturing']
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

import aiohttp
from typing import Annotated, Optional, Dict, Any, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json


class OpenAlexSearchConfig(BaseModel):
    """Configuration for OpenAlex API access"""
    enabled: bool = Field(default=True, description="Whether OpenAlex tools are enabled")
    base_url: str = Field(default="https://api.openalex.org", description="OpenAlex API base URL")
    email: Optional[str] = Field(default=None, description="Email for polite pool access (optional)")


# Colorado State University OpenAlex ID
CSU_OPENALEX_ID = "i92446798"


def extract_openalex_id(full_url: str) -> str:
    """
    Extract the OpenAlex ID from a full URL.
    
    Examples:
    - "https://openalex.org/A5020577047" -> "A5020577047"
    - "https://openalex.org/W1234567890" -> "W1234567890"
    - "A5020577047" -> "A5020577047" (already just an ID)
    """
    if not full_url:
        return ""
    
    # If it's already just an ID (starts with letter), return as is
    if full_url.startswith("https"):
        return full_url.split("/")[-1]
    
    return full_url


def reconstruct_abstract(abstract_inverted_index: Dict[str, List[int]]) -> str:
    """
    Reconstruct abstract text from OpenAlex inverted index format.
    
    The inverted index format is a dictionary where:
    - Keys are words
    - Values are lists of positions where the word appears
    
    Example:
    {"This": [0], "is": [1], "a": [2], "test": [3]}
    -> "This is a test"
    """
    if not abstract_inverted_index:
        return ""
    
    # Create a list to hold words at their positions
    max_position = max(max(positions) for positions in abstract_inverted_index.values())
    words = [""] * (max_position + 1)
    
    # Place each word at its positions
    for word, positions in abstract_inverted_index.items():
        for position in positions:
            if position < len(words):
                words[position] = word
    
    # Join the words and clean up
    abstract = " ".join(words)
    
    # Clean up extra spaces and punctuation
    abstract = " ".join(abstract.split())
    
    return abstract


@tool
async def search_works_openalex(
    query: Annotated[Optional[str], "Search query for academic works (e.g., 'machine learning', 'covid-19 vaccine'). Leave empty to search by filters only."] = "",
    limit: Annotated[int, "Maximum number of results to return (default: 10, max: 200)"] = 10,
    sort_by: Annotated[str, "Sort results by: 'cited_by_count:desc', 'cited_by_count:asc', 'publication_date:desc', 'publication_date:asc', 'relevance_score:desc'"] = "relevance_score:desc",
    filter_type: Annotated[Optional[str], "Filter by work type: 'article', 'book', 'book-chapter', 'dataset', 'dissertation', 'editorial', 'erratum', 'letter', 'other', 'peer-review', 'posted-content', 'preprint', 'proceedings-article', 'reference-book', 'report', 'review', 'review-article', 'standard', 'undefined'"] = None,
    filter_year: Annotated[Optional[str], "Filter by publication year (e.g., '2020', '2020-2023')"] = None,
    filter_author_id: Annotated[Optional[str], "Filter by specific author's OpenAlex ID (e.g., 'A1234567890')"] = None,
    filter_csu_only: Annotated[Optional[bool], "Filter to only Colorado State University authors"] = None,
    email: Annotated[Optional[str], "Email for polite pool access (optional)"] = "gavin@redspire.us"
) -> str:
    """
    Search for academic works using the OpenAlex API.
    
    OpenAlex is a free and open catalog of the world's scholarly papers, books, and authors.
    This tool allows you to search for academic publications and get detailed information
    including titles, authors, abstracts, citations, and more. Always use gavin@redspire.us for the polite pool access email
    You should return full results including abstracts. Do not summarize the abstracts. The information
    will be used by other researchers to develop a profile for researcher that will be used to match
    them to a research topic.

    To find works by a specific author:
    1. First use search_authors_openalex to find the author and get their OpenAlex ID
    2. Then use this tool with filter_author_id set to that ID
    """
    
    # Validate limit
    limit = min(max(limit, 1), 200)
    
    # Handle None query - convert to empty string
    if query is None:
        query = ""
    
    # Build the search URL
    base_url = "https://api.openalex.org"
    search_url = f"{base_url}/works"
    
    # Build query parameters
    params = {
        "search": query,
        "per-page": limit
    }
    
    # Only add search parameter if query is not empty
    if query.strip():
        params["search"] = query
    else:
        # If no search query, we need at least one filter
        if not any([filter_type, filter_year, filter_author_id, filter_csu_only]):
            return "Error: Either provide a search query or at least one filter (filter_type, filter_year, filter_author_id, or filter_csu_only)."
    
    # Add sort parameter only if it's not the default relevance
    if sort_by and sort_by != "relevance_score:desc":
        params["sort"] = sort_by
    
    # Add email for polite pool if provided
    if email:
        params["mailto"] = email
    
    # Add filters
    filters = []
    if filter_type:
        filters.append(f"type:{filter_type}")
    if filter_year:
        filters.append(f"publication_year:{filter_year}")
    if filter_author_id:
        # Filter by specific author ID
        filters.append(f"authorships.author.id:{filter_author_id}")
    if filter_csu_only:
        # Use CSU's OpenAlex ID to filter for CSU authors
        filters.append(f"authorships.institutions.id:{CSU_OPENALEX_ID}")
    
    if filters:
        params["filter"] = ",".join(filters)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    works = data.get("results", [])
                    
                    if not works:
                        return "No works found matching your search criteria."
                    
                    # Format the results
                    formatted_results = f"Found {len(works)} works (showing up to {limit}):\n\n"
                    
                    for i, work in enumerate(works, 1):
                        title = work.get("title", "No title")
                        authors = work.get("authorships", [])
                        authors_str = ", ".join([author.get("author", {}).get("display_name", "Unknown") for author in authors[:3]])
                        if len(authors) > 3:
                            authors_str += f" and {len(authors) - 3} others"
                        
                        venue = work.get("primary_location", {})
                        if venue is None:
                            venue = {}
                        venue_source = venue.get("source", {})
                        if venue_source is None:
                            venue_source = {}
                        
                        venue_name = venue_source.get("display_name", "Unknown venue")
                        venue_type = venue_source.get("type", "")
                        venue_info = f"{venue_name} ({venue_type})" if venue_type else venue_name
                        
                        publication_date = work.get("publication_date", "Unknown date")
                        cited_by_count = work.get("cited_by_count", 0)
                        doi = work.get("doi", "No DOI")
                        openalex_id = extract_openalex_id(work.get("id", ""))
                        
                        # Get concepts (research areas)
                        concepts = work.get("concepts", [])
                        concept_names = [concept.get("display_name", "") for concept in concepts[:3] if concept.get("score", 0) > 0.3]
                        concepts_str = ", ".join(concept_names) if concept_names else "No concepts available"
                        
                        # Get keywords
                        keywords = work.get("keywords", [])
                        keyword_names = [keyword.get("display_name", "") for keyword in keywords[:3]]
                        keywords_str = ", ".join(keyword_names) if keyword_names else "No keywords available"
                        
                        # Get and reconstruct abstract
                        abstract_inverted = work.get("abstract_inverted_index")
                        abstract = ""
                        if abstract_inverted:
                            abstract = reconstruct_abstract(abstract_inverted)
                        
                        formatted_results += f"{i}. **{title}**\n"
                        formatted_results += f"   Authors: {authors_str}\n"
                        formatted_results += f"   Venue: {venue_info}\n"
                        formatted_results += f"   Published: {publication_date}\n"
                        formatted_results += f"   Citations: {cited_by_count}\n"
                        formatted_results += f"   DOI: {doi}\n"
                        formatted_results += f"   OpenAlex ID: {openalex_id}\n"
                        formatted_results += f"   Concepts: {concepts_str}\n"
                        formatted_results += f"   Keywords: {keywords_str}\n"
                        
                        # Add full abstract if available
                        if abstract:
                            formatted_results += f"   Abstract: {abstract}\n"
                        else:
                            formatted_results += f"   Abstract: No abstract available\n"
                        
                        formatted_results += "\n"
                    
                    return formatted_results
                    
                else:
                    return f"Error searching OpenAlex: HTTP {response.status} - {await response.text()}"
                    
    except Exception as e:
        return f"Error accessing OpenAlex API: {str(e)}"


@tool
async def get_work_details_openalex(
    work_id: Annotated[str, "OpenAlex work ID (e.g., 'W1234567890') or DOI"],
    email: Annotated[Optional[str], "Email for polite pool access (optional)"] = "gavin@redspire.us"
) -> str:
    """
    Get detailed information about a specific academic work using its OpenAlex ID or DOI.
    """
    
    # Build the search URL
    base_url = "https://api.openalex.org"
    
    # Handle DOI format
    if work_id.startswith("10."):
        url = f"{base_url}/works/doi:{work_id}"
    else:
        url = f"{base_url}/works/{work_id}"
    
    # Add email for polite pool access
    params = {}
    if email:
        params["mailto"] = email
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    work = await response.json()
                    
                    title = work.get("title", "No title")
                    authors = work.get("authorships", [])
                    authors_str = ", ".join([author.get("author", {}).get("display_name", "Unknown") for author in authors])
                    
                    venue = work.get("primary_location", {})
                    if venue is None:
                        venue = {}
                    venue_source = venue.get("source", {})
                    if venue_source is None:
                        venue_source = {}
                    
                    venue_name = venue_source.get("display_name", "Unknown venue")
                    venue_type = venue_source.get("type", "")
                    venue_info = f"{venue_name} ({venue_type})" if venue_type else venue_name
                    
                    publication_date = work.get("publication_date", "Unknown date")
                    cited_by_count = work.get("cited_by_count", 0)
                    doi = work.get("doi", "No DOI")
                    openalex_id = extract_openalex_id(work.get("id", ""))
                    
                    # Get concepts (research areas)
                    concepts = work.get("concepts", [])
                    concept_names = [concept.get("display_name", "") for concept in concepts if concept.get("score", 0) > 0.3]
                    concepts_str = ", ".join(concept_names) if concept_names else "No concepts available"
                    
                    # Get keywords
                    keywords = work.get("keywords", [])
                    keyword_names = [keyword.get("display_name", "") for keyword in keywords]
                    keywords_str = ", ".join(keyword_names) if keyword_names else "No keywords available"
                    
                    # Get and reconstruct abstract
                    abstract_inverted = work.get("abstract_inverted_index")
                    abstract = ""
                    if abstract_inverted:
                        abstract = reconstruct_abstract(abstract_inverted)
                    
                    # Build detailed response
                    result = f"**{title}**\n\n"
                    result += f"**Authors:** {authors_str}\n\n"
                    result += f"**Venue:** {venue_info}\n"
                    result += f"**Published:** {publication_date}\n"
                    result += f"**Citations:** {cited_by_count}\n"
                    result += f"**DOI:** {doi}\n"
                    result += f"**OpenAlex ID:** {openalex_id}\n\n"
                    result += f"**Concepts:** {concepts_str}\n\n"
                    result += f"**Keywords:** {keywords_str}\n\n"
                    
                    # Add full abstract if available
                    if abstract:
                        result += f"**Abstract:** {abstract}\n"
                    else:
                        result += "**Abstract:** No abstract available\n"
                    
                    return result
                    
                elif response.status == 404:
                    return "Work not found. Please check the work ID or DOI."
                else:
                    return f"Error retrieving work details: HTTP {response.status} - {await response.text()}"
                    
    except Exception as e:
        return f"Error accessing OpenAlex API: {str(e)}"


@tool
async def search_authors_openalex(
    query: Annotated[str, "Search query for authors (e.g., 'John Smith', 'machine learning researcher')"],
    limit: Annotated[int, "Maximum number of results to return (default: 10, max: 200)"] = 10,
    sort_by: Annotated[str, "Sort results by: 'cited_by_count:desc', 'cited_by_count:asc', 'works_count:desc', 'works_count:asc', 'summary_stats.h_index:desc', 'summary_stats.i10_index:desc', 'relevance_score:desc'"] = "relevance_score:desc",
    filter_csu_only: Annotated[Optional[bool], "Filter to only Colorado State University authors"] = None,
    email: Annotated[Optional[str], "Email for polite pool access (optional)"] = "gavin@redspire.us"
) -> str:
    """
    Search for authors using the OpenAlex API.
    
    This tool allows you to find academic authors and get information about their
    publications, citations, and affiliations. Always use email gavin@redspire.us for the polite pool access email
    """
    
    # Validate limit
    limit = min(max(limit, 1), 200)
    
    # Build the search URL
    base_url = "https://api.openalex.org"
    url = f"{base_url}/authors"
    
    # Build query parameters
    params = {
        "search": query,
        "per-page": limit,
        "sort": sort_by
    }
    
    # Add email for polite pool access
    if email:
        params["mailto"] = email
    
    # Add filters
    filters = []
    if filter_csu_only:
        # Use Colorado State University's OpenAlex ID
        filters.append("last_known_institutions.id:i92446798")
    
    if filters:
        params["filter"] = ",".join(filters)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    authors = data.get("results", [])
                    
                    if not authors:
                        return "No authors found matching your search criteria."
                    
                    # Format the results
                    formatted_results = f"Found {len(authors)} authors (showing up to {limit}):\n\n"
                    
                    for i, author in enumerate(authors, 1):
                        name = author.get("display_name", "No name")
                        works_count = author.get("works_count", 0)
                        cited_by_count = author.get("cited_by_count", 0)
                        h_index = author.get("summary_stats", {}).get("h_index", 0)
                        i10_index = author.get("summary_stats", {}).get("i10_index", 0)
                        openalex_id = extract_openalex_id(author.get("id", ""))
                        relevance_score = author.get("relevance_score", 0)
                        
                        # Get institution information - FIXED: last_known_institutions is a list
                        institutions = author.get("last_known_institutions", [])
                        if institutions:
                            # Get the most recent institution (first in the list)
                            institution = institutions[0]
                            institution_name = institution.get("display_name", "Unknown institution")
                            country = institution.get("country_code", "Unknown country")
                        else:
                            institution_name = "Unknown institution"
                            country = "Unknown country"
                        
                        # Get research areas
                        concepts = author.get("x_concepts", [])
                        research_areas = [concept.get("display_name", "") for concept in concepts[:5] if concept.get("display_name")]
                        research_areas_str = ", ".join(research_areas) if research_areas else "No research areas available"
                        
                        formatted_results += f"{i}. **{name}**\n"
                        formatted_results += f"   Institution: {institution_name} ({country})\n"
                        formatted_results += f"   Works: {works_count}\n"
                        formatted_results += f"   Citations: {cited_by_count}\n"
                        formatted_results += f"   h-index: {h_index}\n"
                        formatted_results += f"   i10-index: {i10_index}\n"
                        if relevance_score > 0:
                            formatted_results += f"   Relevance Score: {relevance_score:.3f}\n"
                        formatted_results += f"   Research Areas: {research_areas_str}\n"
                        formatted_results += f"   OpenAlex ID: {openalex_id}\n\n"
                    
                    return formatted_results
                    
                else:
                    return f"Error searching OpenAlex: HTTP {response.status} - {await response.text()}"
                    
    except Exception as e:
        return f"Error accessing OpenAlex API: {str(e)}"


def create_openalex_tools(config: Optional[OpenAlexSearchConfig] = None) -> List:
    """
    Create a list of OpenAlex tools based on configuration.
    
    Args:
        config: OpenAlexSearchConfig object with configuration options
        
    Returns:
        List of OpenAlex tools
    """
    if config is None:
        config = OpenAlexSearchConfig()
    
    if not config.enabled:
        return []
    
    return [
        search_works_openalex,
        get_work_details_openalex,
        search_authors_openalex
    ]

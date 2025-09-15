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


@tool
async def search_works_openalex(
    query: Annotated[str, "Search query for academic works (e.g., 'machine learning', 'covid-19 vaccine')"],
    limit: Annotated[int, "Maximum number of results to return (default: 10, max: 200)"] = 10,
    sort_by: Annotated[str, "Sort results by: 'relevance', 'cited_by_count', 'publication_date'"] = "relevance",
    filter_type: Annotated[Optional[str], "Filter by work type: 'article', 'book', 'book-chapter', 'dataset', 'dissertation', 'editorial', 'erratum', 'letter', 'other', 'peer-review', 'posted-content', 'preprint', 'proceedings-article', 'reference-book', 'report', 'review', 'review-article', 'standard', 'undefined'"] = None,
    filter_year: Annotated[Optional[str], "Filter by publication year (e.g., '2020', '2020-2023')"] = None,
    filter_venue: Annotated[Optional[str], "Filter by venue/journal name"] = None,
    filter_author: Annotated[Optional[str], "Filter by author name"] = None,
    email: Annotated[Optional[str], "Email for polite pool access (optional)"] = None
) -> str:
    """
    Search for academic works using the OpenAlex API.
    
    OpenAlex is a free and open catalog of the world's scholarly papers, books, and authors.
    This tool allows you to search for academic publications and get detailed information
    including titles, authors, abstracts, citations, and more.
    """
    
    # Validate limit
    limit = min(max(limit, 1), 200)
    
    # Build the search URL
    base_url = "https://api.openalex.org"
    search_url = f"{base_url}/works"
    
    # Build query parameters
    params = {
        "search": query,
        "per-page": limit,
        "sort": sort_by
    }
    
    # Add email for polite pool if provided
    if email:
        params["mailto"] = email
    
    # Add filters
    filters = []
    if filter_type:
        filters.append(f"type:{filter_type}")
    if filter_year:
        filters.append(f"publication_year:{filter_year}")
    if filter_venue:
        filters.append(f"venue.display_name:{filter_venue}")
    if filter_author:
        filters.append(f"author.display_name:{filter_author}")
    
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
                        abstract = work.get("abstract_inverted_index", {})
                        cited_by_count = work.get("cited_by_count", 0)
                        publication_date = work.get("publication_date", "Unknown")
                        venue = work.get("primary_location", {}).get("source", {}).get("display_name", "Unknown venue")
                        doi = work.get("doi", "No DOI")
                        openalex_id = work.get("id", "")
                        
                        # Format authors
                        author_names = []
                        for author in authors[:5]:  # Show first 5 authors
                            author_name = author.get("author", {}).get("display_name", "Unknown")
                            author_names.append(author_name)
                        
                        if len(authors) > 5:
                            author_names.append(f"... and {len(authors) - 5} more")
                        
                        authors_str = ", ".join(author_names) if author_names else "Unknown authors"
                        
                        # Format abstract (simplified)
                        abstract_text = "No abstract available"
                        if abstract:
                            # This is a simplified approach - OpenAlex stores abstracts as inverted index
                            # For a full implementation, you'd need to reconstruct the text
                            abstract_text = "Abstract available (inverted index format)"
                        
                        formatted_results += f"{i}. **{title}**\n"
                        formatted_results += f"   Authors: {authors_str}\n"
                        formatted_results += f"   Venue: {venue}\n"
                        formatted_results += f"   Published: {publication_date}\n"
                        formatted_results += f"   Citations: {cited_by_count}\n"
                        formatted_results += f"   DOI: {doi}\n"
                        formatted_results += f"   OpenAlex ID: {openalex_id}\n"
                        formatted_results += f"   Abstract: {abstract_text}\n\n"
                    
                    return formatted_results
                    
                else:
                    return f"Error searching OpenAlex: HTTP {response.status} - {await response.text()}"
                    
    except Exception as e:
        return f"Error accessing OpenAlex API: {str(e)}"


@tool
async def get_work_details_openalex(
    work_id: Annotated[str, "OpenAlex work ID (e.g., 'W1234567890') or DOI"],
    email: Annotated[Optional[str], "Email for polite pool access (optional)"] = None
) -> str:
    """
    Get detailed information about a specific academic work from OpenAlex.
    
    You can provide either an OpenAlex work ID (starts with 'W') or a DOI.
    """
    
    base_url = "https://api.openalex.org"
    
    # Determine if it's a DOI or OpenAlex ID
    if work_id.startswith("10.") or work_id.startswith("http"):
        # It's a DOI
        work_url = f"{base_url}/works/doi:{work_id}"
    elif work_id.startswith("W"):
        # It's an OpenAlex ID
        work_url = f"{base_url}/works/{work_id}"
    else:
        return "Invalid work ID. Please provide either an OpenAlex work ID (starts with 'W') or a DOI."
    
    params = {}
    if email:
        params["mailto"] = email
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(work_url, params=params) as response:
                if response.status == 200:
                    work = await response.json()
                    
                    # Extract detailed information
                    title = work.get("title", "No title")
                    authors = work.get("authorships", [])
                    abstract = work.get("abstract_inverted_index", {})
                    cited_by_count = work.get("cited_by_count", 0)
                    publication_date = work.get("publication_date", "Unknown")
                    venue = work.get("primary_location", {}).get("source", {})
                    doi = work.get("doi", "No DOI")
                    openalex_id = work.get("id", "")
                    concepts = work.get("concepts", [])
                    keywords = work.get("keywords", [])
                    
                    # Format authors with affiliations
                    author_details = []
                    for author in authors:
                        author_name = author.get("author", {}).get("display_name", "Unknown")
                        institutions = author.get("institutions", [])
                        institution_names = [inst.get("display_name", "") for inst in institutions if inst.get("display_name")]
                        affiliation = f" ({', '.join(institution_names)})" if institution_names else ""
                        author_details.append(f"{author_name}{affiliation}")
                    
                    authors_str = "; ".join(author_details) if author_details else "Unknown authors"
                    
                    # Format venue information
                    venue_name = venue.get("display_name", "Unknown venue") if venue else "Unknown venue"
                    venue_type = venue.get("type", "") if venue else ""
                    venue_info = f"{venue_name} ({venue_type})" if venue_type else venue_name
                    
                    # Format concepts
                    concept_names = [concept.get("display_name", "") for concept in concepts[:10] if concept.get("display_name")]
                    concepts_str = ", ".join(concept_names) if concept_names else "No concepts available"
                    
                    # Format keywords
                    keyword_names = [kw.get("display_name", "") for kw in keywords[:10] if kw.get("display_name")]
                    keywords_str = ", ".join(keyword_names) if keyword_names else "No keywords available"
                    
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
                    
                    # Add abstract if available
                    if abstract:
                        result += "**Abstract:** Abstract available (inverted index format - would need reconstruction for full text)\n"
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
    sort_by: Annotated[str, "Sort results by: 'relevance', 'cited_by_count', 'works_count'"] = "relevance",
    filter_institution: Annotated[Optional[str], "Filter by institution name"] = None,
    filter_country: Annotated[Optional[str], "Filter by country code (e.g., 'US', 'GB', 'DE')"] = None,
    email: Annotated[Optional[str], "Email for polite pool access (optional)"] = None
) -> str:
    """
    Search for authors using the OpenAlex API.
    
    This tool allows you to find academic authors and get information about their
    publications, citations, and affiliations.
    """
    
    # Validate limit
    limit = min(max(limit, 1), 200)
    
    # Build the search URL
    base_url = "https://api.openalex.org"
    search_url = f"{base_url}/authors"
    
    # Build query parameters
    params = {
        "search": query,
        "per-page": limit,
        "sort": sort_by
    }
    
    # Add email for polite pool if provided
    if email:
        params["mailto"] = email
    
    # Add filters
    filters = []
    if filter_institution:
        filters.append(f"last_known_institution.display_name:{filter_institution}")
    if filter_country:
        filters.append(f"last_known_institution.country_code:{filter_country}")
    
    if filters:
        params["filter"] = ",".join(filters)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
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
                        openalex_id = author.get("id", "")
                        
                        # Get institution information
                        institution = author.get("last_known_institution", {})
                        institution_name = institution.get("display_name", "Unknown institution")
                        country = institution.get("country_code", "Unknown country")
                        
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
        config: OpenAlex configuration. If None, uses default settings.
        
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

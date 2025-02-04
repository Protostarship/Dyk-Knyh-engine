import pandas as pd
from bs4 import BeautifulSoup
import requests

def get_xpath(element):
    """
    Generates a full XPath from the root <html> element for a given BeautifulSoup element.
    """
    components = []
    while element is not None and element.name is not None:
        sibling_index = 1
        sibling = element
        # Count preceding siblings with the same tag name
        while sibling.previous_sibling:
            sibling = sibling.previous_sibling
            if getattr(sibling, 'name', None) == element.name:
                sibling_index += 1
        components.append(f"{element.name}[{sibling_index}]")
        element = element.parent
    components.reverse()
    return "/" + "/".join(components)

def crawl_dictionary_table(url):
    """
    Crawls the dictionary table from the specified URL.
    Returns a tuple: (data, paths_log, table_info)
      - data: list of extracted rows.
      - paths_log: list of full XPath strings of each element crawled.
      - table_info: list of tuples containing (table element, its full XPath).
    """
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    paths_log = []
    table_info = []
    
    # Log the root <html> element's XPath
    html_element = soup.find('html')
    if html_element:
        html_xpath = get_xpath(html_element)
        paths_log.append(f"Root: {html_xpath}")
    
    # Find and log all table elements
    tables = soup.find_all('table')
    print(f"Found {len(tables)} table(s) on the page.")
    for i, tbl in enumerate(tables):
        tbl_xpath = get_xpath(tbl)
        table_info.append((tbl, tbl_xpath))
        log_entry = f"Table {i} XPath: {tbl_xpath} - ID: {tbl.get('id')}"
        print(log_entry)
        paths_log.append(log_entry)
    
    # Use the table with id 'dictionaryTable' if it exists, else default to the first table
    table = soup.find('table', id='dictionaryTable')
    if not table:
        print("Table with id 'dictionaryTable' not found. Trying the first available table instead.")
        if tables:
            table = tables[0]
        else:
            raise ValueError("No table found on the page")
    
    chosen_table_xpath = get_xpath(table)
    paths_log.append(f"Chosen Table XPath: {chosen_table_xpath}")
    
    # Attempt to get rows with the 'data-index' attribute
    rows = table.find_all('tr', attrs={'data-index': True})
    if not rows:
        msg = "No rows with 'data-index' attribute found. Falling back to <tr> elements within tbody."
        print(msg)
        paths_log.append(msg)
        tbody = table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
        else:
            rows = table.find_all('tr')
            print("No tbody found; falling back to all <tr> elements.")
    
    print(f"Found {len(rows)} row(s) to process.")
    data = []
    for row in rows:
        row_xpath = get_xpath(row)
        paths_log.append(f"Row XPath: {row_xpath}")
        cells = row.find_all('td')
        if len(cells) == 3:
            data.append({
                "Number": cells[0].text.strip(),
                "Origin": cells[1].text.strip(),
                "Translation": cells[2].text.strip()
            })
        else:
            msg = f"Row skipped at XPath {row_xpath} due to unexpected number of <td> elements (found {len(cells)})."
            print(msg)
            paths_log.append(msg)
    
    return data, paths_log, table_info

def write_log(filename, log_lines):
    """
    Writes a list of log lines to the specified file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for line in log_lines:
            f.write(line + "\n")
    print(f"Log written to {filename}")

def save_data(data):
    """
    Saves the extracted data to CSV and Excel files.
    Returns the filenames created.
    """
    df = pd.DataFrame(data)
    csv_file = 'translations.csv'
    excel_file = 'translations.xlsx'
    df.to_csv(csv_file, index=False)
    df.to_excel(excel_file, index=False)
    
    print(f"\nSuccessfully saved {len(df)} entries")
    print("Files created: translations.csv, translations.xlsx")
    return csv_file, excel_file

def main():
    target_url = 'https://basis64-tools.pages.dev/translator'
    try:
        print("Starting data extraction...")
        data, paths_log, table_info = crawl_dictionary_table(target_url)
        
        # If no data is found, do not create CSV/Excel files.
        if not data:
            print("No data found during crawl. Logging error paths.")
            write_log('error_log.txt', paths_log)
            # Log the table details for error investigation.
            table_lines = [f"Table XPath: {xpath} - ID: {tbl.get('id')}" for tbl, xpath in table_info]
            write_log('table_info_error.txt', table_lines)
        else:
            # Save the extracted data.
            csv_file, excel_file = save_data(data)
            # Log all crawled paths for successful crawl.
            write_log('paths_log.txt', paths_log)
            # Log table information.
            table_lines = [f"Table XPath: {xpath} - ID: {tbl.get('id')}" for tbl, xpath in table_info]
            write_log('table_info_success.txt', table_lines)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Optionally log the exception.
        with open('error_log.txt', 'w', encoding='utf-8') as f:
            f.write(str(e))
    
if __name__ == "__main__":
    main()
